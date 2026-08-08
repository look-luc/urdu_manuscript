import torch
import torchvision.io as tv_io
import torchvision.transforms.v2.functional as TVF


class Data_Collector:
    def __init__(self, processor, prompt: str = ""):
        self.processor = processor
        self.prompt = prompt

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
                img_tensor = TVF.pil_to_tensor(img_raw.convert("RGB"))
            else:
                raise ValueError(
                    f"Unsupported image format in collator: {type(img_raw)}"
                )

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

        pad_id = self.processor.tokenizer.pad_token_id
        if pad_id is not None:
            labels[labels == pad_id] = -100

        final_batch["labels"] = labels
        final_batch["user_input_ids"] = user_batch["input_ids"]
        final_batch["user_attention_mask"] = user_batch["attention_mask"]

        return final_batch
