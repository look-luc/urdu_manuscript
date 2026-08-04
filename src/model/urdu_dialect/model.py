import gc
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
    AutoModel,
    AutoProcessor,
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


class unification_urdu_lang_model:
    def __init__(
        self,
        model_id: str = "OpenGVLab/InternVL2-8B",
        prompt: str = """
            You are an expert multilingual OCR system specializing in high-accuracy transcription of Arabic, Urdu (including Nastaliq and Naskh scripts), and Persian text. Analyze the image carefully and transcribe the text line-by-line from right to left, maintaining the original paragraph breaks and line structure.
            Output ONLY the raw extracted text. Do not fix spelling mistakes, do not normalize text structure, do not add translations, and do not include any conversational filler, notes, or markdown explanations before or after the transcription.
        """,
        batch_size: int = 64,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
        torch.device(self.device)

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
        if self.device != "cuda":
            raise ValueError("CUDA device not detected")
        if self.device == "cuda":
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
            torch.cuda.reset_peak_memory_stats()

            torch.backends.cudnn.enabled = False
            torch.backends.cudnn.benchmark = False

        config = AutoConfig.from_pretrained(
            self.model_id, trust_remote_code=True
        )
        config.use_cache = False

        model = AutoModel.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            load_in_8bit=True,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map={"": self.device},
        )

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "k_proj",
                "v_proj",
                "o_proj",
                "gate_proj",
                "up_proj",
                "down_proj",
                "wqkv",
                "wo",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.print_trainable_parameters()

        data = get_datasets()

        return model, processor, data

    def train(self):
        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            gradient_accumulation_steps=8,
            gradient_checkpointing=True,
            num_train_epochs=1,
            dataloader_num_workers=0,
            learning_rate=2e-5,
            max_steps=2500,
            eval_strategy="steps",
            eval_steps=500,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=1.0,
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

        save_path = "../text_extraction/urdu_model/saved_model"
        trainer.save_model(save_path)
        self.processor.save_pretrained(save_path)

        return train_result
