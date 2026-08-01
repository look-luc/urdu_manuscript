import os
import urllib.request

import numpy as np
import torch
import torchvision.io as torchvision_io
from torchvision.io import ImageReadMode


def _fetch_url_bytes(url: str, timeout: int = 10) -> bytes:
    """Downloads raw bytes from an HTTP/HTTPS image URL."""
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


class QwenDataCollator:
    def __init__(self, processor, prompt: str):
        self.processor = processor
        self.prompt = prompt

    def _extract_text_string(self, val) -> str:
        if isinstance(val, str):
            return val
        if isinstance(val, list):
            for item in val:
                res = self._extract_text_string(item)
                if res:
                    return res
        elif isinstance(val, dict):
            for key in ["text", "content", "target"]:
                if key in val:
                    return self._extract_text_string(val[key])
        return str(val) if val is not None else ""

    def _load_image_tensor(self, raw_img) -> torch.Tensor | None:
        if isinstance(raw_img, dict):
            if "bytes" in raw_img and raw_img["bytes"]:
                byte_tensor = torch.frombuffer(
                    bytearray(raw_img["bytes"]), dtype=torch.uint8
                )
                return torchvision_io.decode_image(byte_tensor, mode=ImageReadMode.RGB)
            elif "path" in raw_img and raw_img["path"]:
                path_str = str(raw_img["path"])
                if path_str.startswith(("http://", "https://")):
                    url_bytes = _fetch_url_bytes(path_str)
                    byte_tensor = torch.frombuffer(
                        bytearray(url_bytes), dtype=torch.uint8
                    )
                    return torchvision_io.decode_image(byte_tensor, mode=ImageReadMode.RGB)
                return torchvision_io.read_image(path_str, mode=ImageReadMode.RGB)

        elif isinstance(raw_img, str):
            if raw_img.startswith(("http://", "https://")):
                url_bytes = _fetch_url_bytes(raw_img)
                byte_tensor = torch.frombuffer(
                    bytearray(url_bytes), dtype=torch.uint8
                )
                return torchvision_io.decode_image(byte_tensor, mode=ImageReadMode.RGB)
            return torchvision_io.read_image(raw_img, mode=ImageReadMode.RGB)

        elif isinstance(raw_img, torch.Tensor):
            return raw_img

        elif hasattr(raw_img, "__array__"):
            return torch.from_numpy(np.asarray(raw_img))

        return None

    def __call__(self, features):
        text_str = []
        imgs = []

        for feature in features:
            raw_img = feature.get("image") if "image" in feature else feature.get("images")
            if raw_img is None:
                continue

            try:
                img_tensor = self._load_image_tensor(raw_img)
            except Exception:
                continue

            if img_tensor is None:
                continue

            # Enforce (3, H, W) layout
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3, 4]:
                img_tensor = img_tensor.permute(2, 0, 1)

            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            elif img_tensor.shape[0] == 4:
                img_tensor = img_tensor[:3, :, :]

            if img_tensor.dtype != torch.uint8:
                img_tensor = img_tensor.to(torch.uint8)

            # Convert directly to channel-last uint8 NumPy array (H, W, 3) in memory
            img_numpy = img_tensor.permute(1, 2, 0).cpu().numpy()

            target_text = self._extract_text_string(feature.get("text"))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_numpy},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_text},
                    ],
                },
            ]

            # In-memory numpy array guarantees exact matching with processor(images=...)
            formatted_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            imgs.append(img_numpy)
            text_str.append(formatted_text)

        if len(imgs) == 0:
            dummy_np = np.full((28, 28, 3), 255, dtype=np.uint8)

            dummy_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": dummy_np},
                        {"type": "text", "text": self.prompt},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": ""},
                    ],
                },
            ]

            dummy_text = self.processor.apply_chat_template(
                dummy_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            imgs.append(dummy_np)
            text_str.append(dummy_text)

        batch = self.processor(
            text=text_str,
            images=imgs,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
