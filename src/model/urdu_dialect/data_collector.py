import numpy as np
import torch
import torchvision.io as io
from torchvision.io import ImageReadMode


def _ensure_min_dimensions_tensor(
    img: torch.Tensor, min_dim: int = 28, max_aspect_ratio: float = 4.0
) -> torch.Tensor:
    c, h, w = img.shape
    aspect_ratio = max(w / h, h / w) if h > 0 and w > 0 else 1.0

    if w >= min_dim and h >= min_dim and aspect_ratio <= max_aspect_ratio:
        return img

    side = max(w, h, min_dim)
    canvas = torch.full(
        (c, side, side), 255, dtype=img.dtype, device=img.device
    )
    offset_x = (side - w) // 2
    offset_y = (side - h) // 2
    canvas[:, offset_y : offset_y + h, offset_x : offset_x + w] = img
    return canvas


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

    def __call__(self, features):
        text_str = []
        imgs = []

        for feature in features:
            raw_img = feature.get("image") if "image" in feature else feature.get("images")
            if raw_img is None:
                continue

            img_tensor = None

            # 1. Decode to torch.Tensor without PIL
            if isinstance(raw_img, dict):
                if "bytes" in raw_img and raw_img["bytes"]:
                    byte_tensor = torch.frombuffer(
                        bytearray(raw_img["bytes"]), dtype=torch.uint8
                    )
                    img_tensor = io.decode_image(byte_tensor, mode=ImageReadMode.RGB)
                elif "path" in raw_img and raw_img["path"]:
                    img_tensor = io.read_image(str(raw_img["path"]), mode=ImageReadMode.RGB)
            elif isinstance(raw_img, str):
                img_tensor = io.read_image(raw_img, mode=ImageReadMode.RGB)
            elif isinstance(raw_img, torch.Tensor):
                img_tensor = raw_img
            elif hasattr(raw_img, "__array__"):
                # Handles PIL or NumPy inputs by converting directly to Torch Tensor
                img_tensor = torch.from_numpy(np.asarray(raw_img))

            if img_tensor is None:
                continue

            # 2. Enforce 3D (C, H, W) layout
            if img_tensor.ndim == 2:
                img_tensor = img_tensor.unsqueeze(0)
            elif img_tensor.ndim == 3 and img_tensor.shape[2] in [1, 3, 4]:
                img_tensor = img_tensor.permute(2, 0, 1)

            # 3. Standardize channels to 3-channel RGB uint8
            if img_tensor.shape[0] == 1:
                img_tensor = img_tensor.repeat(3, 1, 1)
            elif img_tensor.shape[0] == 4:
                img_tensor = img_tensor[:3, :, :]

            if img_tensor.dtype != torch.uint8:
                img_tensor = img_tensor.to(torch.uint8)

            # 4. Canvas padding for minimum dimension
            img_tensor = _ensure_min_dimensions_tensor(
                img_tensor, min_dim=28, max_aspect_ratio=4.0
            )

            # Convert (3, H, W) Tensor -> (H, W, 3) uint8 NumPy Array for HuggingFace Processor
            img_numpy = img_tensor.permute(1, 2, 0).cpu().numpy()

            target_text = self._extract_text_string(feature.get("text"))

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": self.prompt},
                        {"type": "image"},
                    ],
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": target_text},
                    ],
                },
            ]

            formatted_text = self.processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=False,
            )

            imgs.append(img_numpy)
            text_str.append(formatted_text)

        batch = self.processor(
            text=text_str,
            images=imgs if len(imgs) > 0 else None,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
