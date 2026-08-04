import torch
import torchvision.io as tv_io
from torchvision.transforms.functional import to_pil_image


class Data_Collector:
    def __init__(self, processor, prompt: str = ""):
        self.processor = processor
        self.prompt = prompt
        self.pad_token_id = self.processor.tokenizer.pad_token_id

    def __call__(self, features):
        images = []
        full_text_prompts = []
        user_prompt_lengths = []

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
            else:
                raise ValueError(
                    f"Unsupported image format in collator: {type(img_raw)}"
                )

            pil_img = to_pil_image(img_tensor)
            images.append(pil_img)
            txt_content = feature.get("text", "")

            user_messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": pil_img},
                        {"type": "text", "text": self.prompt},
                    ],
                }
            ]

            full_messages = user_messages + [
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": txt_content},
                    ],
                }
            ]

            # Format user prompt to get exact prompt token length
            user_prompt_text = self.processor.apply_chat_template(
                user_messages, tokenize=False, add_generation_prompt=True
            )
            full_text = self.processor.apply_chat_template(
                full_messages, tokenize=False, add_generation_prompt=False
            )

            # Measure prompt length in tokens
            prompt_token_ids = self.processor.tokenizer.encode(user_prompt_text)
            user_prompt_lengths.append(len(prompt_token_ids))
            full_text_prompts.append(full_text)

        batch = self.processor(
            text=full_text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()

        # Mask prompt tokens and padding tokens explicitly
        for i, prompt_len in enumerate(user_prompt_lengths):
            labels[i, :prompt_len] = -100

        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels

        return batch
