import os
import sys
from pathlib import Path

import evaluate
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

SCRATCH_BASE = Path(f"/scratch/alpine/{os.getenv('USER', 'lude4390')}")
DEFAULT_OUTPUT_DIR = SCRATCH_BASE / "model" / "urdu_manuscript_model"
DEFAULT_RESULTS_DIR = SCRATCH_BASE / "results"


class AutoregressiveTrainer(Trainer):
    """Custom Trainer overriding prediction_step for Side 2 autoregressive generation."""
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
                generated_ids = unwrapped_model.generate(
                    input_ids=inputs["user_input_ids"],
                    attention_mask=inputs["user_attention_mask"],
                    pixel_values=inputs.get("pixel_values"),
                    image_grid_thw=inputs.get("image_grid_thw"),
                    max_new_tokens=128,
                    use_cache=True,
                )
            else:
                generated_ids = None

        labels = inputs["labels"]
        return (loss, generated_ids, labels)


class unification_urdu_lang_model:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-3B-Instruct",
        prompt: str = """You are an expert OCR model for historical Urdu and Arabic-script manuscripts with expert knowledge of Farsi/Persian, Arabic and Urdu. Transcribe the text line-by-line. If there are marginal notes or footnotes, transcribe them separately at the end under 'Marginalia'. Do not translate.""",
        batch_size: int = 64,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt
        self.batch_size = batch_size

        self.model, self.processor, self.data = self._setup()

    def _compute_metrics(eval_pred, tokenizer):
        predictions, labels = eval_pred

        if isinstance(predictions, tuple):
            predictions = predictions[0]

        if predictions.ndim == 3:
            predictions = np.argmax(predictions, axis=-1)

        decoded_preds = []
        decoded_labels = []

        ignore_index = -100
        pad_id = tokenizer.pad_token_id

        for pred_seq, label_seq in zip(predictions, labels):
            clean_pred = [
                int(token) for token in pred_seq
                if token != ignore_index and token != pad_id and token >= 0
            ]
            clean_label = [
                int(token) for token in label_seq
                if token != ignore_index and token != pad_id and token >= 0
            ]

            decoded_preds.append(tokenizer.decode(clean_pred, skip_special_tokens=True))
            decoded_labels.append(tokenizer.decode(clean_label, skip_special_tokens=True))

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

        if hasattr(processor.image_processor, "max_pixels"):
            processor.image_processor.max_pixels = 512 * 28 * 28
            processor.image_processor.min_pixels = 256 * 28 * 28

        peft_config = LoraConfig(
            r=64,
            lora_alpha=64,
            target_modules=[
                "q_proj", "v_proj", "k_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj",
                "merger.mlp.0", "merger.mlp.2",
                "qkv", "proj",
            ],
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
        )

        model = get_peft_model(model, peft_config)
        model.enable_input_require_grads()

        return model, processor, data

    def train(self, output_dir: str = str(DEFAULT_OUTPUT_DIR)):
        train_dataset = self.data["train"]
        test_dataset = self.data["test"]

        eval_subset = test_dataset.select(range(min(50, len(test_dataset))))

        training_args = TrainingArguments(
            output_dir=str(DEFAULT_RESULTS_DIR),
            per_device_train_batch_size=4,
            per_device_eval_batch_size=4,
            gradient_accumulation_steps=4,
            gradient_checkpointing=True,
            gradient_checkpointing_kwargs={"use_reentrant": False},
            dataloader_num_workers=4,
            dataloader_pin_memory=True,
            dataloader_persistent_workers=True,
            max_steps=1000,
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=250,
            save_strategy="steps",
            save_steps=250,
            save_total_limit=1,
            learning_rate=2E-4,
            bf16=True,
            remove_unused_columns=False,
            max_grad_norm=1.0,
            warmup_steps=100,
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
        )

        train_result = trainer.train()
        os.makedirs(output_dir, exist_ok=True)
        trainer.save_model(output_dir)
        self.processor.save_pretrained(output_dir)

        return train_result
