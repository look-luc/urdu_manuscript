import math
import urllib.request

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.io as torchvision_io
from qwen_vl_utils import process_vision_info
from torchvision.io import ImageReadMode
from torchvision.transforms.functional import to_pil_image


def _fetch_url_bytes(url: str, timeout: int = 10) -> bytes:
    """Downloads raw bytes from an HTTP/HTTPS image URL."""
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


class QwenDataCollator:
    def __init__(self, processor, prompt: str, min_pixels: int, max_pixels: int):
        self.processor = processor
        self.prompt = prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, features):
        text_str = []
        imgs = []

        for feature in features:
            raw_img = feature.get("image") if "image" in feature else feature.get("images")
            raw_txt = feature.get("text") if "text" in feature
            if raw_img is None:
                continue

            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": raw_img,
                            "min_pixels": self.min_pixels,
                            "max_pixels": self.max_pixels,
                        },
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": raw_txt}],
                },
            ]

            image_inputs, _ = process_vision_info(messages)

            formatted_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if image_inputs:
                imgs.extend(image_inputs)
            else:
                imgs.append(raw_img)

            text_str.append(formatted_text)

        batch = self.processor(
            text=text_str,
            images=imgs,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
