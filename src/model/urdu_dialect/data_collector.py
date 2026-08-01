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
    req = urllib.request.Request(
        url, headers={"User-Agent": "Mozilla/5.0"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read()


class QwenDataCollator:
    def __init__(self, processor, prompt: str, min_pixels: int, max_pixels: int):
        self.processor = processor
        self.prompt = prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

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

    def _pad_img_ten(self, img):
        channels = img.shape[0]
        h = img.shape[1]
        w = img.shape[2]

        min_edge = max(56, int(math.sqrt(self.min_pixels)))
        target_h = max(h, min_edge)
        target_w = max(w, min_edge)

        if target_h % 28 != 0:
            target_h += 28 - (target_h % 28)
        if target_w % 28 != 0:
            target_w += 28 - (target_w % 28)

        pad_h = target_h - h
        pad_w = target_w - w

        if pad_h > 0 or pad_w > 0:
            top = pad_h // 2
            bottom = pad_h - top
            left = pad_w // 2
            right = pad_w - left
            img = F.pad(img, (left, right, top, bottom))
        return img

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

            img_tensor = self._pad_img_ten(img_tensor)

            pil_img = to_pil_image(img_tensor)

            target_text = self._extract_text_string(feature.get("text"))

            img_content = {
                "type": "image",
                "image": pil_img,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }

            text_content = {
                "type": "text",
                "text": self.prompt,
            }

            user_message = {
                "role": "user",
                "content": [img_content, text_content],
            }

            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": target_text}],
            }

            messages = [user_message, assistant_message]

            image_inputs, _ = process_vision_info(messages)

            formatted_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if image_inputs:
                imgs.extend(image_inputs)
            else:
                imgs.append(pil_img)

            text_str.append(formatted_text)

        if len(imgs) == 0:
            dummy_tensor = torch.full((3, 56, 56), 255, dtype=torch.uint8)
            dummy_pil = to_pil_image(dummy_tensor)

            img_content = {
                "type": "image",
                "image": dummy_pil,
                "min_pixels": self.min_pixels,
                "max_pixels": self.max_pixels,
            }

            text_content = {
                "type": "text",
                "text": self.prompt,
            }

            user_message = {
                "role": "user",
                "content": [img_content, text_content],
            }

            assistant_message = {
                "role": "assistant",
                "content": [{"type": "text", "text": ""}],
            }

            dummy_messages = [user_message, assistant_message]

            dummy_image_inputs, _ = process_vision_info(dummy_messages)

            dummy_text = self.processor.apply_chat_template(
                dummy_messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            if dummy_image_inputs:
                imgs.extend(dummy_image_inputs)
            else:
                imgs.append(dummy_pil)

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
