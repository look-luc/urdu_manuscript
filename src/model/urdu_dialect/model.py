import sys
from pathlib import Path

import torch
from data_collector import Data_Collector
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

root_dir = Path(__file__).resolve().parents[3]
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data import get_data


class unification_urdu_lang_model:
    def __init__(
        self,
        model_id:str="Qwen/Qwen2.5-VL-72B-Instruct",
        prompt:str="""
            You are a multilingual OCR model. You must extract the text given to you.
        """
    )->None:
        torch.backends.cudnn.enabled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id=model_id
        self.model, self.processor, self.data = self._setup()

        self.prompt = prompt

    def _setup (self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
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

        processed_arabic_test = arabic_data_test.map(self._process, remove_columns=arabic_data_test.column_names)
        processed_urdu_test = urdu_data_test.map(self._process, remove_columns=urdu_data_test.column_names)
        processed_farsi_test = farsi_data_test.map(self._process, remove_columns=farsi_data_test.column_names)
