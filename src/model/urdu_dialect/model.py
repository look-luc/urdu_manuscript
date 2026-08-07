import gc
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torchmetrics.functional.text import bleu_score
from transformers import (
    AutoConfig,
    AutoModelForMultimodalLM,
    AutoProcessor,
    BitsAndBytesConfig,
    Trainer,
    TrainingArguments,
)

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data.get_data import get_datasets

from .data_collector import Data_Collector

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

ALLOCATED_CPU = os.environ.get('SLURM_CPUS_PER_TASK')

def preprocess_logits_for_metrics(logits, labels):
    """
    Reduces 3D output logits (Batch, Seq_Len, Vocab_Size) to 2D token IDs (Batch, Seq_Len)
    directly on the GPU before cross-batch evaluation collection.
    """
    if isinstance(logits, tuple):
        logits = logits[0]
    return logits.argmax(dim=-1)

class unification_urdu_lang_model:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        prompt: str = """You are an expert OCR model for historical Urdu and Arabic-script manuscripts. Transcribe the text line-by-line. If there are marginal notes or footnotes, transcribe them separately at the end under 'Marginalia'. Do not translate.""",
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.batch_size = batch_size

        self.model, self.processor, self.data = self._setup()

    def _compute_metrics(self, eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        decoded_preds = []
        decoded_labels = []

        for i in range(len(label_ids)):
            valid_mask = label_ids[i] != -100
            valid_label_tokens = label_ids[i][valid_mask]
            valid_pred_tokens = pred_ids[i][valid_mask]

            pred_str = self.processor.tokenizer.decode(
                valid_pred_tokens, skip_special_tokens=True
            ).strip()
            label_str = self.processor.tokenizer.decode(
                valid_label_tokens, skip_special_tokens=True
            ).strip()

            decoded_preds.append(pred_str if pred_str else " ")
            decoded_labels.append(label_str if label_str else " ")

        cer_score = cer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        wer_score = wer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        bleu_targets = [[label] for label in decoded_labels]
        try:
            bleu_score_val = bleu_score(decoded_preds, bleu_targets).item()
        except Exception:
            bleu_score_val = 0.0

        return {"CER": cer_score, "WER": wer_score, "BLEU": bleu_score_val}

    def _setup(self):
        data = get_datasets()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        torch.device(self.device)

        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

        config = AutoConfig.from_pretrained(self.model_id)
        config.use_cache = False

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = AutoModelForMultimodalLM.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map={"": self.device},
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa"
        )

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = 512 * 28 * 28
            processor.image_processor.min_pixels = 256 * 28 * 28

        peft_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "merger.mlp.0", "merger.mlp.2"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()

        return model, processor, data

    def train(self, output_dir: str = "./model/urdu_manuscript_model"):
        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        eval_subset = test_dataset.select(range(min(200, len(test_dataset))))

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            max_steps=500,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            save_total_limit=1,
            learning_rate=5e-5,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_subset,
            data_collator=Data_Collector(
                self.processor,
                prompt=self.prompt,
            ),
            compute_metrics=self._compute_metrics,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        return train_result
