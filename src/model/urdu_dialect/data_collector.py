import os

import requests
import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as F


class Data_Collector:

    def __init__(self, processor, prompt: str, image_base_dir: str = ""):
        self.processor = processor
        self.prompt = prompt
        self.image_base_dir = image_base_dir
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.assistant_start_token = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def _is_valid_header(self, bytes_data):
        if len(bytes_data) < 4:
            return False
        if (
            bytes_data[0] == 0xFF
            and bytes_data[1] == 0xD8
            and bytes_data[2] == 0xFF
        ):
            return True
        elif (
            bytes_data[0] == 0x89
            and bytes_data[1] == 0x50
            and bytes_data[2] == 0x4E
            and bytes_data[3] == 0x47
        ):
            return True
        return False

    def _load_image(self, image_input):
        if hasattr(image_input, "convert"):
            return image_input.convert("RGB")

        if isinstance(image_input, dict):
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
                return F.to_pil_image(image_tensor)

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
                    return F.to_pil_image(image_tensor)
                else:
                    if not os.path.isabs(image_path) and self.image_base_dir:
                        image_path = os.path.join(
                            self.image_base_dir, image_path
                        )
                    image_tensor = tv_io.read_image(
                        image_path, mode=tv_io.ImageReadMode.RGB
                    )
                    return F.to_pil_image(image_tensor)

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
                return F.to_pil_image(image_tensor)
            else:
                if not os.path.isabs(image_path) and self.image_base_dir:
                    image_path = os.path.join(self.image_base_dir, image_path)
                image_tensor = tv_io.read_image(
                    image_path, mode=tv_io.ImageReadMode.RGB
                )
                return F.to_pil_image(image_tensor)

        return None

    def _prep_image(self, image_pil):
        """Ensures canvas is at least 112x112 so Qwen's grid_h and grid_w stay >= 2."""
        if image_pil is None:
            return None

        w, h = image_pil.size
        min_dim = 112
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

        return image_pil

    def __call__(self, features):
        features = [f for f in features if f is not None]
        if not features:
            raise ValueError("Data_Collector received an empty batch.")

        images_list = []
        formatted_texts = []

        for feature in features:
            raw_pil = self._load_image(feature.get("image"))
            image_pil = self._prep_image(raw_pil)
            if image_pil is None:
                continue

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
                    "content": [{"type": "text", "text": feature["text"]}],
                },
            ]

            text_str = self.processor.apply_chat_template(
                message, tokenize=False, add_generation_prompt=False
            )

            images_list.append(image_pil)
            formatted_texts.append(text_str)

        if not images_list:
            raise ValueError("All samples in batch failed image loading.")

        inputs = self.processor(
            text=formatted_texts,
            images=images_list,
            padding=True,
            min_pixels=256 * 28 * 28,
            max_pixels=512 * 28 * 28,
            return_tensors="pt",
        )

        labels = inputs["input_ids"].clone()
        for i in range(len(formatted_texts)):
            row_labels = labels[i]
            for row in range(
                len(row_labels) - len(self.assistant_start_token) + 1
            ):
                if (
                    row_labels[
                        row : row + len(self.assistant_start_token)
                    ].tolist()
                    == self.assistant_start_token
                ):
                    labels[i, : row + len(self.assistant_start_token)] = -100
                    break
        labels[inputs["input_ids"] == self.pad_token_id] = -100

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "labels": labels,
            "pixel_values": inputs["pixel_values"].bfloat16(),
            "image_grid_thw": inputs["image_grid_thw"],
        }
