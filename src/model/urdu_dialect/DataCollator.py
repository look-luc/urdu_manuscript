class QwenDataCollator:
    def __init__(self, processor):
        self.processor = processor

    def __call__(self, features):
        batch = self.processor(
            images=[feature["image"] for feature in features],
            text=[feature["text"] for feature in features],
            padding=True,
            return_tensors="pt",
        )
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        batch["labels"] = labels

        return batch
