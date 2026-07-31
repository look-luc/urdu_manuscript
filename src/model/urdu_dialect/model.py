import gc
import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import requests
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as F
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
        self.model, self.processor, self.data = self._setup()

        self.prompt = prompt
        self.batch_size = batch_size

    def _compute_metrics(self, eval_pred):
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids

        if isinstance(pred_ids, tuple):
            pred_ids = pred_ids[0]

        # Convert logits of shape (batch, seq_len, vocab_size) to token IDs
        if pred_ids.ndim == 3:
            pred_ids = np.argmax(pred_ids, axis=-1)

        pad_id = self.processor.tokenizer.pad_token_id

        # Mask out ignored tokens (-100) using label_ids alignment
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
            self.model_id, min_pixels=256 * 28 * 28, max_pixels=512 * 28 * 28
        )

        data = get_datasets()

        for split in data.keys():
            cols = getattr(data[split], "column_names", None)
            if cols is not None and "image" in cols:
                data[split] = data[split].cast_column("image", HFImage())

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

    def _load_image(self, example):
        """Helper method to load the image consistently for filtering and processing."""
        image_input = example.get("image")
        image_pil = None

        if hasattr(image_input, "save"):
            image_pil = image_input.convert("RGB")

        elif isinstance(image_input, dict):
            if image_input.get("bytes") is not None:
                raw_bytes = image_input["bytes"]
                if not self._is_valid_header(raw_bytes):
                    return None
                storage_tensor = torch.frombuffer(
                    bytearray(raw_bytes), dtype=torch.uint8
                )
                image_tensor = tv_io.decode_image(
                    storage_tensor, mode=tv_io.ImageReadMode.RGB
                )
                image_pil = F.to_pil_image(image_tensor)

            elif image_input.get("path") is not None:
                image_path = image_input["path"]
                if image_path.startswith(("http://", "https://")):
                    response = requests.get(image_path, timeout=10)
                    if response.status_code != 200:
                        return None
                    storage_tensor = torch.frombuffer(
                        bytearray(response.content), dtype=torch.uint8
                    )
                    image_tensor = tv_io.decode_image(
                        storage_tensor, mode=tv_io.ImageReadMode.RGB
                    )
                    image_pil = F.to_pil_image(image_tensor)
                else:
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(IMAGE_BASE_DIR, image_path)
                    if not self._is_valid_file(image_path):
                        return None
                    image_tensor = tv_io.read_image(
                        image_path, mode=tv_io.ImageReadMode.RGB
                    )
                    image_pil = F.to_pil_image(image_tensor)

        elif isinstance(image_input, str):
            image_path = image_input
            if image_path.startswith(("http://", "https://")):
                response = requests.get(image_path, timeout=10)
                if response.status_code != 200:
                    return None
                storage_tensor = torch.frombuffer(
                    bytearray(response.content), dtype=torch.uint8
                )
                image_tensor = tv_io.decode_image(
                    storage_tensor, mode=tv_io.ImageReadMode.RGB
                )
                image_pil = F.to_pil_image(image_tensor)
            else:
                if not os.path.isabs(image_path):
                    image_path = os.path.join(IMAGE_BASE_DIR, image_path)
                if not self._is_valid_file(image_path):
                    return None
                image_tensor = tv_io.read_image(
                    image_path, mode=tv_io.ImageReadMode.RGB
                )
                image_pil = F.to_pil_image(image_tensor)

        return image_pil

    def _process(self, example):
        try:
            image_pil = self._load_image(example)
            if image_pil is None:
                raise ValueError("Failed to load image tensor")

            w, h = image_pil.size
            min_dim = 56
            max_aspect = 8.0

            target_w = max(w, min_dim)
            target_h = max(h, min_dim)

            if target_w / target_h > max_aspect:
                target_h = int(target_w / max_aspect)
            elif target_h / target_w > max_aspect:
                target_w = int(target_h / max_aspect)

            pad_w = max(0, target_w - w)
            pad_h = max(0, target_h - h)

            if pad_w > 0 or pad_h > 0:
                padding = [
                    pad_w // 2,
                    pad_h // 2,
                    pad_w - (pad_w // 2),
                    pad_h - (pad_h // 2),
                ]
                image_pil = F.pad(image_pil, padding=padding, fill=255)

        except Exception:
            return {
                "input_ids": torch.zeros((1,), dtype=torch.long),
                "attention_mask": torch.zeros((1,), dtype=torch.long),
                "pixel_values": torch.zeros((1, 1176), dtype=torch.float32),
                "image_grid_thw": torch.tensor([[1, 2, 2]], dtype=torch.long),
                "is_valid": False,
            }

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

        formatted_text = self.processor.apply_chat_template(
            message, tokenize=False, add_generation_prompt=False
        )

        inputs = self.processor(
            text=[formatted_text],
            images=[image_pil],
            padding=True,
            max_length=1024,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
            return_tensors="pt",
        )

        pixel_values = inputs["pixel_values"]
        while pixel_values.dim() > 2:
            pixel_values = pixel_values.squeeze(0)
        if pixel_values.dim() == 1:
            pixel_values = pixel_values.unsqueeze(0)

        grid_thw = inputs["image_grid_thw"]
        while grid_thw.dim() > 2:
            grid_thw = grid_thw.squeeze(0)
        if grid_thw.dim() == 1:
            grid_thw = grid_thw.unsqueeze(0)

        grid_h = grid_thw[0][1].item()
        grid_w = grid_thw[0][2].item()

        is_valid = True
        if grid_h < 2 or grid_w < 2 or grid_h % 2 != 0 or grid_w % 2 != 0:
            is_valid = False

        return {
            "input_ids": inputs["input_ids"].squeeze(0),
            "attention_mask": inputs["attention_mask"].squeeze(0),
            "pixel_values": pixel_values,
            "image_grid_thw": grid_thw,
            "is_valid": is_valid,
        }

    def train(self):
        self.max_tokens = 2000

        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        train_cols = getattr(train_dataset, "column_names", None)
        test_cols = getattr(test_dataset, "column_names", None)

        processed_train = train_dataset.map(self._process, remove_columns=train_cols)
        processed_test = test_dataset.map(self._process, remove_columns=test_cols)

        processed_train = processed_train.filter(lambda x: x["is_valid"])
        processed_test = processed_test.filter(lambda x: x["is_valid"])

        try:
            next(iter(processed_train))
            print("Successfully verified active stream for processed_train.")
        except StopIteration:
            raise ValueError("processed_train iterator is empty!")

        data_collector = Data_Collector(processor=self.processor)

        self.model.enable_input_require_grads()
        self.model.gradient_checkpointing_enable()

        training_args = Seq2SeqTrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=2,
            per_device_eval_batch_size=2,
            gradient_accumulation_steps=4,
            num_train_epochs=1,
            dataloader_num_workers=4,
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
            train_dataset=processed_train,
            eval_dataset=processed_test,
            data_collator=data_collector,
            compute_metrics=self._compute_metrics,
        )

        train_result = trainer.train()

        save_path = "../text_extraction/urdu_model/saved_model"
        trainer.save_model(save_path)
        self.processor.save_pretrained(save_path)

        return train_result
