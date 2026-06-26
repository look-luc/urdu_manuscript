import sys
from pathlib import Path

import evaluate
import numpy as np
import torch
from datasets import concatenate_datasets
from peft import LoraConfig, get_peft_model
from torchmetrics.functional.text import bleu_score
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
    Trainer,
    TrainingArguments,
)

from data import get_data

from .data_collector import Data_Collector

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

cer_metric = evaluate.load("cer")
wer_metric = evaluate.load("wer")

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

        # Converts raw logits into the most likely Token IDs
        # logits shape: (batch_size, sequence_length, vocab_size)
        pred_ids = np.argmax(logits, axis=-1)

        decoded_preds = []
        decoded_labels = []

        shift_logits = logits[:, :-1, :]
        shift_labels = label_ids[:, 1:]

        pred_ids = np.argmax(shift_logits, axis=-1)
        # Clean and decode row by row
        for i in range(len(label_ids)):
            # Isolates the text the assistant was supposed to output (ignoring -100)
            valid_indices = shift_labels[i] != -100

            row_label_ids = shift_labels[i][valid_indices]
            row_pred_ids = pred_ids[i][valid_indices]

            # Convert token IDs back to human-readable strings using the processor
            pred_text = self.processor.tokenizer.decode(row_pred_ids, skip_special_tokens=True)
            label_text = self.processor.tokenizer.decode(row_label_ids, skip_special_tokens=True)

            decoded_preds.append(pred_text)
            decoded_labels.append(label_text)

        # Metric Comparison Loop
        tp, fp, fn = 0, 0, 0
        for pred, target in zip(decoded_preds, decoded_labels):
            p_arr = np.array(pred.split())
            t_arr = np.array(target.split())

            unique_tokens = np.union1d(p_arr, t_arr)

            for token in unique_tokens:
                pred_count = np.sum(p_arr == token)
                target_count = np.sum(t_arr == token)

                tp += min(pred_count, target_count)

                fp += max(0, pred_count - target_count)
                fn += max(0, target_count - pred_count)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        f1_score = (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

        targets = [[label] for label in decoded_labels]
        bleu_score_ocr = bleu_score(decoded_preds, targets, n_gram=4)

        cer_score = cer_metric.compute(predictions=decoded_preds, references=decoded_labels)
        wer_score = wer_metric.compute(predictions=decoded_preds, references=decoded_labels)

        return {
            "F1": f1_score,
            "BLEU score": bleu_score_ocr,
            "CER score": cer_score,
            "WER score": wer_score
        }

    def _setup (self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map=self.device,
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.model_id)

        data = get_data.get_datasets()

        return  model, processor, data

    def _process(self, example):
        self.image = example["image"]
        self.text = example["text"]

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": self.image,
                        "min_pixels": 512 * 512,
                        "max_pixels": 14 * 14 * 1024 * 1024
                    },
                    {
                        "type": "text",
                        "text": self.prompt
                    }
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "text",
                        "text": self.text
                    }
                ]
            }
        ]

        text_prompt = self.processor.apply_chat_template(
            message,
            tokenize=False,
            return_assistant_tokens_mask=True,
            add_generation_prompt=False
        )

        inputs = self.processor(
            text=[text_prompt],
            images=[self.image],
            padding=False,
            return_tensors='pt'
        )

        return {
            "input_ids": inputs["input_ids"][0],
            "attention_mask": inputs["attention_mask"][0],
            "pixel_values": inputs["pixel_values"],
            "image_grid_thw": inputs["image_grid_thw"]
        }

    def train(self):
        self.max_tokens = 2000

        arabic_data_train = self.data["train"]["arabic"]
        urdu_data_train = self.data["train"]["urdu"]
        farsi_data_train = self.data["train"]["farsi"]

        arabic_data_test = self.data["test"]["arabic"]
        urdu_data_test = self.data["test"]["urdu"]
        farsi_data_test = self.data["test"]["farsi"]

        processed_arabic_train = arabic_data_train.map(self._process, remove_columns=arabic_data_train.column_names)
        processed_urdu_train = urdu_data_train.map(self._process, remove_columns=urdu_data_train.column_names)
        processed_farsi_train = farsi_data_train.map(self._process, remove_columns=farsi_data_train.column_names)

        train_data = concatenate_datasets(
            [
                processed_arabic_train,
                processed_urdu_train,
                processed_farsi_train
            ]
        )

        processed_arabic_test = arabic_data_test.map(self._process, remove_columns=arabic_data_test.column_names)
        processed_urdu_test = urdu_data_test.map(self._process, remove_columns=urdu_data_test.column_names)
        processed_farsi_test = farsi_data_test.map(self._process, remove_columns=farsi_data_test.column_names)

        test_data = concatenate_datasets(
            [
                processed_arabic_test,
                processed_urdu_test,
                processed_farsi_test
            ]
        )

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
            per_device_train_batch_size=1,      # Minimize active batch memory footprint
            gradient_accumulation_steps=4,      # Simulates a batch size of 4 safely
            gradient_checkpointing=True,        # Crucial OOM mitigation flag
            bf16=True,                          # Ensures bfloat16 math optimization
            optim="adamw_torch_fused",          # Memory-efficient execution choice
            remove_unused_columns=False,        # MANDATORY: do not drop visual columns
            learning_rate=2e-5,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=200,
        )

        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=data_collector,
            compute_metrics=self._compute_metrics,
        )

        return trainer.train()
