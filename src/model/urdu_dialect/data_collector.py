from qwen_vl_utils import process_vision_info


class QwenDataCollator:
    def __init__(self, processor, prompt: str, min_pixels: int, max_pixels: int):
        self.processor = processor
        self.prompt = prompt
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels

    def __call__(self, features):
        text_str = []
        imgs = []
        prompt_lens = []

        for feature in features:
            if not feature or not isinstance(feature, dict):
                continue

            raw_img = feature.get("image") if "image" in feature else feature.get("images")
            raw_txt = feature.get("text") if "text" in feature else ""

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

            prompt_messages = [messages[0]]
            prompt_text = self.processor.apply_chat_template(
                prompt_messages, tokenize=False, add_generation_prompt=True
            )
            prompt_tokens = self.processor.tokenizer(prompt_text, return_tensors="pt")["input_ids"]
            prompt_lens.append(prompt_tokens.shape[1])

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

        if not text_str or not imgs:
            return {}

        batch = self.processor(
            text=text_str,
            images=imgs,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
            padding=True,
            return_tensors="pt",
        )

        if batch is None:
            return {}

        labels = batch["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id

        for idx, p_len in enumerate(prompt_lens):
            labels[idx, :p_len] = -100

        labels[labels == pad_id] = -100
        batch["labels"] = labels

        return batch
