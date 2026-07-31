import urllib.request

import torch
import torchvision.io as io
from torchvision.io import ImageReadMode
from torchvision.transforms.functional import to_pil_image


def _is_pil_image_by_module(obj):
    if obj is None or isinstance(obj, dict):
        return False

    obj_type = type(obj)
    module_name = getattr(obj_type, "__module__", "")

    if module_name and module_name.startswith("PIL"):
        return True

    return False


def _is_pil_img_duck_typing(obj):
    if obj is None or isinstance(obj, dict):
        return False

    return (
        hasattr(obj, "convert")
        and hasattr(obj, "size")
        and hasattr(obj, "mode")
    )


class QwenDataCollator:
    def __init__(self, processor, prompt):
        self.processor = processor
        self.prompt = prompt

    def _extract_text_string(self, val):
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

    def _fetch_url_bytes(self, url: str) -> bytes:
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "Mozilla/5.0"}
        )
        with urllib.request.urlopen(req) as response:
            return response.read()

    def _process_image_path_or_url(self, path_str: str):
        if path_str.startswith(("http://", "https://")):
            url_bytes = self._fetch_url_bytes(path_str)
            byte_tensor = torch.frombuffer(bytearray(url_bytes), dtype=torch.uint8)
            img = to_pil_image(io.decode_image(
                byte_tensor, mode=ImageReadMode.RGB
            ))
        else:
            img = to_pil_image(io.read_image(
                path_str, mode=ImageReadMode.RGB
            ))
        return img.convert("RGB")

    def _has_image_content(self, raw_txt):
        if isinstance(raw_txt, list):
            for msg in raw_txt:
                if isinstance(msg, dict) and isinstance(msg.get("content"), list):
                    for item in msg["content"]:
                        if isinstance(item, dict) and item.get("type") == "image":
                            return True
        return False

    def __call__(self, features):
        text_str = []
        imgs = []

        for feature in features:
            raw_img = (
                feature.get("image")
                if "image" in feature
                else feature.get("images")
            )
            if raw_img is None:
                continue

            img_obj = None
            if isinstance(raw_img, dict):
                if "bytes" in raw_img and raw_img["bytes"]:
                    byte_tensor = torch.frombuffer(
                        bytearray(raw_img["bytes"]), dtype=torch.uint8
                    )
                    img_obj = to_pil_image(io.decode_image(
                        byte_tensor,
                        mode=ImageReadMode.RGB
                    )).convert("RGB")
                elif "path" in raw_img and raw_img["path"]:
                    img_obj = self._process_image_path_or_url(str(raw_img["path"]))
            elif isinstance(raw_img, str):
                img_obj = self._process_image_path_or_url(raw_img)
            else:
                img_obj = raw_img

            if img_obj is None:
                continue

            if _is_pil_image_by_module(img_obj) or _is_pil_img_duck_typing(img_obj):
                img_obj = img_obj.convert("RGB")
            elif isinstance(img_obj, torch.Tensor) or hasattr(img_obj, "__array__"):
                img_obj = to_pil_image(img_obj).convert("RGB")
            else:
                continue

            raw_txt = feature.get("text")

            if self._has_image_content(raw_txt):
                formatted_text = self.processor.apply_chat_template(
                    raw_txt,
                    tokenize=False,
                    add_generation_prompt=False,
                )
            else:
                target_text = self._extract_text_string(raw_txt)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
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
                formatted_text = self.processor.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=False,
                )

            imgs.append([img_obj])
            text_str.append(formatted_text)

        batch = self.processor(
            images=imgs,
            text=text_str,
            padding=True,
            return_tensors="pt",
        )

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
