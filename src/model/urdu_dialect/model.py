import os
import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
import torchvision.io as tv_io  # Added for native tensor image decoding
from evaluate import load
from peft import LoraConfig, get_peft_model
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
        model_id:str="Qwen/Qwen2.5-VL-7B-Instruct",
        prompt:str="""
            You are an expert multilingual OCR system specializing in high-accuracy transcription of Arabic, Urdu (including Nastaliq and Naskh scripts), and Persian text.
            Analyze the image carefully and transcribe the text line-by-line from right to left, maintaining the original paragraph breaks and line structure.
            Output ONLY the raw extracted text. Do not fix spelling mistakes, do not normalize text structure, do not add translations, and do not include any conversational filler, notes, or markdown explanations before or after the transcription.
        """
    )->None:
        torch.backends.cudnn.enabled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id=model_id
        self.model, self.processor, self.data = self._setup()

        self.prompt = prompt

    def _compute_metrics(self, eval_pred):
        logits, label_ids = eval_pred.predictions
        if isinstance(logits, tuple):
            logits = logits[0]

        pred_ids = np.argmax(logits, axis=-1)

        decoded_preds = self.processor.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)

        clean_label_ids = np.where(
            label_ids != -100,
            label_ids,
            self.processor.tokenizer.pad_token_id
        )
        decoded_labels = self.processor.tokenizer.batch_decode(
            clean_label_ids,
            skip_special_tokens=True
        )

        decoded_preds = [pred.strip() if pred.strip() else " " for pred in decoded_preds]
        decoded_labels = [label.strip() if label.strip() else " " for label in decoded_labels]

        cer_score = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        wer_score = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        bleu_targets = [[label] for label in decoded_labels]

        try:
            bleu_score_val = bleu_score(decoded_preds, bleu_targets).item()
        except Exception:
            bleu_score_val = 0.0

        return {
            "CER": cer_score,
            "WER": wer_score,
            "BLEU": bleu_score_val
        }

    def _setup (self):
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"
        )
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            quantization_config=quantization_config,
            attn_implementation="sdpa"
        )
        processor = AutoProcessor.from_pretrained(self.model_id, min_pixels=256*256, max_pixels=512*512)

        data = get_datasets()

        return  model, processor, data

    def _process(self, example):
        try:
            image_input = example["image"]
            image_tensor = None

            if isinstance(image_input, dict):
                if image_input.get("bytes") is not None:
                    raw_bytes = image_input["bytes"]
                    storage_tensor = torch.frombuffer(raw_bytes, dtype=torch.uint8)
                    image_tensor = tv_io.decode_image(storage_tensor, mode=tv_io.ImageReadMode.RGB)
                elif image_input.get("path") is not None:
                    image_path = image_input["path"]
                    if not os.path.isabs(image_path):
                        image_path = os.path.join(IMAGE_BASE_DIR, image_path)
                    if os.path.exists(image_path):
                        image_tensor = tv_io.read_image(image_path, mode=tv_io.ImageReadMode.RGB)

            elif isinstance(image_input, str):
                image_path = image_input
                if not os.path.isabs(image_path):
                    image_path = os.path.join(IMAGE_BASE_DIR, image_path)
                if os.path.exists(image_path):
                    image_tensor = tv_io.read_image(image_path, mode=tv_io.ImageReadMode.RGB)

            if image_tensor is None:
                return {"is_valid": False}

            # Standard Hugging Face vision processors process raw tensors assuming Channels-Last order
            # to avoid misinterpreting width/height dimensions as color channels.
            image_tensor = image_tensor.permute(1, 2, 0)

            text = example["text"]
            message = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": text}
                    ]
                }
            ]

            text_prompt = self.processor.apply_chat_template(
                message,
                tokenize=False,
                add_generation_prompt=False
            )

            inputs = self.processor(
                text=[text_prompt],
                images=[image_tensor],
                padding=False,
                return_tensors='pt'
            )

            inputs_dict = {}
            for k, v in inputs.items():
                if k in ["input_ids", "attention_mask"] and isinstance(v, torch.Tensor):
                    inputs_dict[k] = v.squeeze(0)
                else:
                    inputs_dict[k] = v

            inputs_dict["is_valid"] = True
            return inputs_dict

        except Exception as e:
            print(f"Failed to process sample: {e}")
            return {"is_valid": False}

    def train(self):
        self.max_tokens = 2000

        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        # Filter out invalid rows using the returned boolean dictionary flag
        processed_train = train_dataset.map(self._process).filter(lambda example: example.get("is_valid", False))
        processed_test = test_dataset.map(self._process).filter(lambda x: x.get("is_valid", False))

        data_collector = Data_Collector(processor=self.processor)

        peft_config = LoraConfig(
            r=16,
            lora_alpha=32,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM"
        )

        self.model = get_peft_model(self.model, peft_config) #type: ignore

        training_args = TrainingArguments(
            output_dir="./results",
            ignore_data_skip=True,
            per_device_train_batch_size=1,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            bf16=True,
            optim="adamw_torch_fused",
            remove_unused_columns=False,
            learning_rate=2e-5,
            logging_steps=10,
            max_steps=5000,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
            dataloader_num_workers=0,
            dataloader_pin_memory=False,
            accelerator_config={
                "dispatch_batches": False,
                "split_batches": False
            },
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_train,
            eval_dataset=processed_test,
            data_collator=data_collector,
            compute_metrics=self._compute_metrics,
        )

        return trainer.train()
