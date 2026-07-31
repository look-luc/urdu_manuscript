class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        text_str = []
        imgs = []

        for feature in features:
            if "image" in feature:
                imgs.append(feature.get("image"))
            elif "images" in feature:
                imgs.append(feature.get("images"))

            raw_txt = feature.get("text")

            if isinstance(raw_txt, list) and all(isinstance(item, dict) for item in raw_txt):
                formated_text = self.processor.apply_chat_template(
                    raw_txt,
                    tokenize=False,
                    add_generation_prompt=False
                )
            elif isinstance(raw_txt, dict):
                key = next(
                    (
                        k for k in ["text", "content", "target"] if k in raw_txt
                    ),
                    None
                )
                if key is not None:
                    formated_text = raw_txt[key]
                else:
                    formated_text = ""
            else:
                formated_text = raw_txt

            text_str.append(formated_text)

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
