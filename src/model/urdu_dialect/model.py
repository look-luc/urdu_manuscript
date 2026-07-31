import gc
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import Image as HFImage
from evaluate import load
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchmetrics.functional.text import bleu_score
from transformers import (
    AutoConfig,
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data.get_data import IMAGE_BASE_DIR, get_datasets

from .data_collector import Data_Collector

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")
f1_metric = load("f1")


class unification_urdu_lang_model:

    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = """
            You are an expert multilingual OCR system specializing in high-accuracy transcription of Arabic, Urdu (including Nastaliq and Naskh scripts), and Persian text.
            Analyze the image carefully and transcribe the text line-by-line from right to left, maintaining the original paragraph breaks and line structure.
            Output ONLY the raw extracted text. Do not fix spelling mistakes, do not normalize text structure, do not add translations, and do not include any conversational filler, notes, or markdown explanations before or after the transcription.
        """,
        batch_size: int = 64,
    ) -> None:
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

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

        pad_id = self.processor.tokenizer.pad_token_id

        clean_pred_ids = np.where(label_ids != -100, pred_ids, pad_id)
        clean_label_ids = np.where(label_ids != -100, label_ids, pad_id)

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

        try:
            bleu_score_val = bleu_score(decoded_preds, bleu_targets).item()
        except Exception:
            bleu_score_val = 0.0

        return {"CER": cer_score, "WER": wer_score, "BLEU": bleu_score_val}

    def _setup(self):
        torch.backends.cudnn.enabled = False
        torch.backends.cudnn.benchmark = False

        if self.device != "cuda":
            raise ValueError("CUDA device not detected")

        torch.cuda.empty_cache()
        gc.collect()

        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )

        config = AutoConfig.from_pretrained(self.model_id)
        config.use_cache = False

        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            config=config,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation="sdpa",
        )
        model.config.use_cache = False

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
        )

        data = get_datasets()

        for split in data.keys():
            cols = getattr(data[split], "column_names", None)
            if cols is not None and "image" in cols:
                data[split] = data[split].cast_column("image", HFImage())

        return model, processor, data

    def train(self):
        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        data_collector = Data_Collector(
            processor=self.processor,
            prompt=self.prompt,
            image_base_dir=IMAGE_BASE_DIR,
        )

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            dataloader_num_workers=0,
            learning_rate=2e-5,
            max_steps=2500,
            eval_strategy="steps",
            eval_steps=500,
            predict_with_generate=False,
            generation_max_length=512,
            bf16=True,
            remove_unused_columns=False,
        )

        trainer = Seq2SeqTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            data_collator=data_collector,
            compute_metrics=self._compute_metrics,
        )

        train_result = trainer.train()

        save_path = "../text_extraction/urdu_model/saved_model"
        trainer.save_model(save_path)
        self.processor.save_pretrained(save_path)

        return train_result
