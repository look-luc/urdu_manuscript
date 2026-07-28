import gc
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as F
from evaluate import load
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from torchmetrics.functional.text import bleu_score
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2_5_VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
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
        self.model, self.processor, self.data = self._setup()

        self.prompt = prompt
        self.batch_size = batch_size

    def _compute_metrics(self, eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        clean_label_ids = np.where(
            label_ids != -100, label_ids, self.processor.tokenizer.pad_token_id
        )
        clean_pred_ids = np.where(
            pred_ids != -100, pred_ids, self.processor.tokenizer.pad_token_id
        )

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
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            quantization_config=bnb_config,
            device_map={"": 0},
            attn_implementation="sdpa",
        )

        model = prepare_model_for_kbit_training(model)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)

        processor = AutoProcessor.from_pretrained(
            self.model_id, min_pixels=128 * 128, max_pixels=256 * 256
        )

        data = get_datasets()

        return model, processor, data

    def _is_valid_header(self, bytes_data):
        if len(bytes_data) < 4:
            return False
        if bytes_data[0] == 0xFF and bytes_data[1] == 0xD8 and bytes_data[2] == 0xFF:
            return True
        elif (
            bytes_data[0] == 0x89
            and bytes_data[1] == 0x50
            and bytes_data[2] == 0x4E
            and bytes_data[3] == 0x47
        ):
            return True
        return False

    def _is_valid_file(self, path: str) -> bool:
        try:
            if not os.path.exists(path):
                return False
            try:
                with open(path, "rb") as f:
                    header = f.read(4)
                return self._is_valid_header(header)
            except Exception:
                return False
        except Exception:
            return False

    def _process(self, example):
        image_input = example["image"]
        image_tensor = None
        if isinstance(image_input, dict):
            if image_input.get("bytes") is not None:
                raw_bytes = image_input["bytes"]
                if not self._is_valid_header(raw_bytes):
                    return {"is_valid": False}
                storage_tensor = torch.frombuffer(
                    bytearray(raw_bytes), dtype=torch.uint8
                )
                image_tensor = tv_io.decode_image(
                    storage_tensor, mode=tv_io.ImageReadMode.RGB
                )
            elif image_input.get("path") is not None:
                image_path = image_input["path"]
                if not os.path.isabs(image_path):
                    image_path = os.path.join(IMAGE_BASE_DIR, image_path)
                if self._is_valid_file(image_path):
                    image_tensor = tv_io.read_image(
                        image_path, mode=tv_io.ImageReadMode.RGB
                    )

        elif isinstance(image_input, str):
            image_path = image_input
            if not os.path.isabs(image_path):
                image_path = os.path.join(IMAGE_BASE_DIR, image_path)
            if self._is_valid_file(image_path):
                image_tensor = tv_io.read_image(
                    image_path, mode=tv_io.ImageReadMode.RGB
                )

        if image_tensor is None:
            return {"is_valid": False}

        image_tensor = F.to_pil_image(image_tensor)

        text = example["text"]
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image"},
                    {"type": "text", "text": self.prompt},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "text", "text": example["text"]}],
            },
        ]

        text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text],
            images=[image_tensor],
            padding=False,
            truncation=True,
            max_length=384,
            min_pixels = 128 * 128,
            max_pixels = 200 * 200,
            return_tensors="pt",
        )

        input_dict = {}
        input_dict["input_ids"] = inputs["input_ids"].squeeze(0)
        input_dict["attention_mask"] = inputs["attention_mask"].squeeze(0)
        input_dict["pixel_values"] = inputs["pixel_values"]
        input_dict["image_grid_thw"] = inputs["image_grid_thw"]
        input_dict["is_valid"] = True
        return input_dict

    def _preprocess_eval_logits(self, logits, labels):
        if isinstance(logits, tuple):
            logits = logits[0]
        return torch.argmax(logits, dim=-1)

    def train(self):
        self.max_tokens = 2000

        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        processed_train = train_dataset.map(self._process).filter(
            lambda example: example.get("is_valid", False)
        )
        processed_test = test_dataset.map(self._process).filter(
            lambda x: x.get("is_valid", False)
        )

        data_collector = Data_Collector(processor=self.processor)

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        training_args = TrainingArguments(
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            output_dir="./results",
            ignore_data_skip=True,
            per_device_train_batch_size=1,
            per_device_eval_batch_size=1,
            eval_accumulation_steps=1,
            gradient_accumulation_steps=8,
            bf16=True,
            optim="paged_adamw_8bit",
            remove_unused_columns=False,
            learning_rate=2e-5,
            logging_steps=10,
            max_steps=2000,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=250,
            accelerator_config={
                "dispatch_batches": False,
                "split_batches": False,
            },
            gradient_checkpointing_kwargs={
                "use_reentrant": False
            }
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_test,
            data_collator=data_collector,
            preprocess_logits_for_metrics=self._preprocess_eval_logits,
            compute_metrics=self._compute_metrics,
        )

        return trainer.train()
