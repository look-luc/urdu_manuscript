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
    EarlyStoppingCallback,
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

SCRATCH_BASE = Path(f"/projects/{os.getenv('USER', 'lude4390')}/urdu_manuscript")
DEFAULT_OUTPUT_DIR = SCRATCH_BASE / "model" / "urdu_manuscript_model"


class AutoregressiveTrainer(Trainer):
    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: dict,
        prediction_loss_only: bool,
        ignore_keys=None,
    ):
        if prediction_loss_only:
            return super().prediction_step(
                model, inputs, prediction_loss_only, ignore_keys=ignore_keys
            )

        inputs = self._prepare_inputs(inputs)
        unwrapped_model = self.accelerator.unwrap_model(model)

        with torch.no_grad():
            outputs = model(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                pixel_values=inputs.get("pixel_values"),
                image_grid_thw=inputs.get("image_grid_thw"),
                labels=inputs["labels"],
            )
            loss = outputs.loss.detach()

            if "user_input_ids" in inputs:
                prompt_len = inputs["user_input_ids"].shape[1]

                proc = getattr(self, "processing_class", None)
                if proc is not None and hasattr(proc, "tokenizer") and proc.tokenizer is not None:
                    tokenizer = proc.tokenizer
                elif proc is not None:
                    tokenizer = proc
                else:
                    tokenizer = getattr(unwrapped_model, "tokenizer", None)

                if tokenizer is None:
                    raise ValueError(
                        "Tokenizer could not be resolved. Ensure `processing_class` is passed to AutoregressiveTrainer."
                    )

                pad_id = (
                    tokenizer.pad_token_id
                    if tokenizer.pad_token_id is not None
                    else tokenizer.eos_token_id
                )

                generated_ids = unwrapped_model.generate(
                    input_ids=inputs["user_input_ids"],
                    attention_mask=inputs["user_attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=512,
                    repetition_penalty=1.1,
                    no_repeat_ngram_size=0,
                    eos_token_id=pad_id,
                    use_cache=True,
                )

                # Slice off prompt
                generated_ids = generated_ids[:, prompt_len:]

                # Pad to fixed 512 length across batches
                pad_len = 512 - generated_ids.shape[1]
                if pad_len > 0:
                    generated_ids = torch.nn.functional.pad(
                        generated_ids, (0, pad_len), value=pad_id
                    )
                elif pad_len < 0:
                    generated_ids = generated_ids[:, :512]
            else:
                generated_ids = None

        labels = inputs["labels"]
        return (loss, generated_ids, labels)


class unification_urdu_lang_model:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        prompt: str = """Transcribe the text in this historical manuscript image. Output only the exact transcribed text.""",
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.batch_size = batch_size

        self.model, self.processor, self.data = self._setup()

    def _compute_metrics(self, eval_pred):
        predictions, labels = eval_pred

        if predictions is None:
            return {"CER": 0.0, "WER": 0.0, "BLEU": 0.0}

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = []
        decoded_labels = []

        ignore_index = -100
        tokenizer = self.processor.tokenizer
        pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else -1

        for pred_seq, label_seq in zip(predictions, labels):
            clean_pred = [
                int(token)
                for token in pred_seq
                if token != ignore_index and token != pad_id and token >= 0
            ]
            clean_label = [
                int(token)
                for token in label_seq
                if token != ignore_index and token != pad_id and token >= 0
            ]

            decoded_preds.append(
                tokenizer.decode(clean_pred, skip_special_tokens=True).strip()
            )
            decoded_labels.append(
                tokenizer.decode(clean_label, skip_special_tokens=True).strip()
            )

        if not decoded_preds or not any(decoded_preds):
            return {"CER": 0.0, "WER": 0.0, "BLEU": 0.0}

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

        if torch.cuda.is_available():
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
            dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            attn_implementation="sdpa",
        )

        processor = AutoProcessor.from_pretrained(
            self.model_id,
            trust_remote_code=True,
        )

        if hasattr(processor, "tokenizer") and processor.tokenizer is not None:
            processor.tokenizer.padding_side = "left"

        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = (896-128) * 28 * 28
            processor.image_processor.min_pixels = 256 * 28 * 28

        peft_config = LoraConfig(
            r=32,
            lora_alpha=32,
            target_modules=[
                "q_proj",
                "v_proj",
                "k_proj",
                "o_proj",
                "lm_head",
                "embed_tokens"
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()
        model.tokenizer = processor.tokenizer
        return model, processor, data

    def train(self, output_dir: str = str(DEFAULT_OUTPUT_DIR)):
        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        eval_subset = test_dataset.select(range(min(100, len(test_dataset))))

        training_args = TrainingArguments(
            output_dir="./results",
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=2,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=2,
            dataloader_pin_memory=True,
            # num_train_epochs=1,
            max_steps=1500,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=None,
            load_best_model_at_end=True,
            metric_for_best_model="CER",
            greater_is_better=False,
            learning_rate=2e-5,
            weight_decay=0.01,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=0.3,
            warmup_steps=50,
            lr_scheduler_type="cosine",
            optim="paged_adamw_8bit",
        )

        trainer = AutoregressiveTrainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_subset,
            data_collator=Data_Collector(
                self.processor,
                prompt=self.prompt,
            ),
            compute_metrics=self._compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=4)],
        )

        train_result = trainer.train()

        os.makedirs(output_dir, exist_ok=True)

        trainer.save_model(output_dir)
        trainer.save_state()
        trainer.save_metrics("train", train_result.metrics)

        self.processor.save_pretrained(output_dir)

        if (
            hasattr(self.processor, "tokenizer")
            and self.processor.tokenizer is not None
        ):
            self.processor.tokenizer.save_pretrained(output_dir)

        base_model = (
            self.model.get_base_model()
            if hasattr(self.model, "get_base_model")
            else self.model
        )
        gen_config = getattr(base_model, "generation_config", None)
        if gen_config is not None and hasattr(gen_config, "save_pretrained"):
            gen_config.save_pretrained(output_dir)

        return train_result
