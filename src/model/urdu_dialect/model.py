import gc
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from torchmetrics.functional.text import bleu_score
from torchmetrics.text import EditDistance
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    LlavaNextForConditionalGeneration,
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
f1_metric = EditDistance()

ALLOCATED_CPU = os.environ.get('SLURM_CPUS_PER_TASK')

class unification_urdu_lang_model:
    def __init__(
        self,
        model_id: str = "llava-hf/llama3-llava-next-8b-hf",
        prompt: str = """...""",
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

        pad_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

        clean_label_ids = np.where(label_ids != -100, label_ids, pad_id)
        clean_pred_ids = np.where(label_ids != -100, pred_ids, pad_id)

        decoded_preds = self.processor.tokenizer.batch_decode(
            clean_pred_ids, skip_special_tokens=True
        )
        decoded_labels = self.processor.tokenizer.batch_decode(
            clean_label_ids, skip_special_tokens=True
        )

        decoded_preds = [
            pred.strip() if pred.strip() else " " for pred in decoded_preds
        ]
        decoded_labels = [
            label.strip() if label.strip() else " " for label in decoded_labels
        ]

        cer_score = cer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )
        wer_score = wer_metric.compute(
            predictions=decoded_preds, references=decoded_labels
        )

        bleu_targets = [[label] for label in decoded_labels]

        f1_metric.update(decoded_preds, decoded_labels)
        f1_score = f1_metric.compute()
        f1_metric.reset()

        try:
            bleu_score_val = bleu_score(decoded_preds, bleu_targets).item()
        except Exception:
            bleu_score_val = 0.0

        return {"F1": f1_score, "CER": cer_score, "WER": wer_score, "BLEU": bleu_score_val}

    def _setup(self):
        data = get_datasets()

        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch.device(self.device)

        if self.device != "cuda":
            raise ValueError("CUDA device not detected")

        _ = torch.zeros(1, device=self.device)

        gc.collect()
        torch.cuda.empty_cache()

        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = False

        config = AutoConfig.from_pretrained(self.model_id)
        config.use_cache = False

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        model = LlavaNextForConditionalGeneration.from_pretrained(
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

        peft_config = LoraConfig(
            r=64,
            lora_alpha=32,
            target_modules=[
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "linear_1", "linear_2"
            ],
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        return model, processor, data

    def train(self, output_dir: str = "./model/urdu_manuscript_model"):
        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=8,
            dataloader_pin_memory=False,
            gradient_checkpointing=True,
            dataloader_num_workers=0,
            dataloader_persistent_workers=False,
            num_train_epochs=1,
            learning_rate=1e-4,
            max_steps=2500,
            eval_strategy="steps",
            eval_steps=500,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=0.5,
            warmup_steps=125,
            lr_scheduler_type="cosine",
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=Data_Collector(
                self.processor,
                prompt=self.prompt,
            ),
            compute_metrics=self._compute_metrics,
        )

        train_result = trainer.train()
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        return train_result
