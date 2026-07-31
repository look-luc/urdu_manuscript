import torch
import torchvision.io as io
from torchvision.io import ImageReadMode


class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

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

            if isinstance(raw_img, dict):
                if "bytes" in raw_img and raw_img["bytes"]:
                    byte_tensor = torch.frombuffer(
                        raw_img["bytes"], dtype=torch.uint8
                    )
                    img_obj = io.decode_image(
                        byte_tensor,
                        mode=ImageReadMode.RGB
                    ).permute(1, 2, 0)
                elif "path" in raw_img and raw_img["path"]:
                    img_obj = io.read_image(
                        str(raw_img["path"]),
                        mode=ImageReadMode.RGB
                    ).permute(1, 2, 0)
                else:
                    img_obj = raw_img
            elif isinstance(raw_img, str):
                img_obj = io.read_image(
                    raw_img,
                    mode=ImageReadMode.RGB
                ).permute(1, 2, 0)
            else:
                img_obj = raw_img

            raw_txt = feature.get("text")

            if isinstance(raw_txt, list) and all(
                isinstance(item, dict) for item in raw_txt
            ):
                formatted_text = self.processor.apply_chat_template(
                    raw_txt, tokenize=False, add_generation_prompt=False
                )
            elif isinstance(raw_txt, dict):
                if "role" in raw_txt:
                    formatted_text = self.processor.apply_chat_template(
                        [raw_txt], tokenize=False, add_generation_prompt=False
                    )
                else:
                    formatted_text = self._extract_text_string(raw_txt)
            else:
                formatted_text = str(raw_txt) if raw_txt is not None else ""

            imgs.append(img_obj)
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
