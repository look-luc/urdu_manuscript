import torch
import torchvision.io as tv_io
import torchvision.transforms.functional as F


def pad_to_min_dim(img_tensor: torch.Tensor, min_dim: int = 336) -> torch.Tensor:
    c, h, w = img_tensor.shape
    new_h = max(h, min_dim)
    new_w = max(w, min_dim)

    padded_tensor = torch.full(
        (c, new_h, new_w),
        fill_value=255,
        dtype=img_tensor.dtype,
        device=img_tensor.device,
    )

    top = (new_h - h) // 2
    left = (new_w - w) // 2

    padded_tensor[:, top : top + h, left : left + w] = img_tensor
    return padded_tensor


class Data_Collector:
    def __init__(self, processor, prompt: str = ""):
        self.processor = processor
        self.prompt = prompt
        self.pad_token_id = (
            self.processor.tokenizer.pad_token_id
            if self.processor.tokenizer.pad_token_id is not None
            else self.processor.tokenizer.eos_token_id
        )

    def __call__(self, features):
        images = []
        full_text_prompts = []
        user_text_prompts = []

        for feature in features:
            img_raw = feature["image"]

            if isinstance(img_raw, bytes):
                byte_buffer = torch.frombuffer(
                    bytearray(img_raw), dtype=torch.uint8
                )
                img_tensor = tv_io.decode_image(
                    byte_buffer, mode=tv_io.ImageReadMode.RGB
                )
            elif isinstance(img_raw, torch.Tensor):
                img_tensor = img_raw
            elif hasattr(img_raw, "convert"):
                img_tensor = F.pil_to_tensor(img_raw.convert("RGB"))
            else:
                raise ValueError(
                    f"Unsupported image format in collator: {type(img_raw)}"
                )

            img_tensor = pad_to_min_dim(img_tensor, min_dim=336)
            images.append(img_tensor)
            txt_content = feature.get("text", "")

            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            full_messages = [
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
                        {"type": "text", "text": txt_content},
                    ],
                },
            ]

            user_prompt_text = self.processor.apply_chat_template(
                user_messages, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, add_generation_prompt=False
            )

            user_text_prompts.append(user_prompt_text)
            full_text_prompts.append(full_text)

        full_batch = self.processor(
            text=full_text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        user_batch = self.processor.tokenizer(
            text=user_text_prompts,
            padding=True,
            return_tensors="pt",
        )

        final_batch = full_batch.copy()
        labels = full_batch["input_ids"].clone()

        batch_size = labels.size(0)
        for i in range(batch_size):
            prompt_len = user_batch["attention_mask"][i].sum().item()
            labels[i, :prompt_len] = -100

        labels[labels == self.pad_token_id] = -100
        final_batch["labels"] = labels

        return final_batch
