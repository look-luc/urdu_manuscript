class Data_Collector:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.assistant_start_token = self.processor.tokenizer.encode("<|im_start|>assistant\n", add_special_tokens=False)

    def __call__(self, features):# features is a list of dicts returned by the dataset's _process function
        # Separates the text tensors from visual features
        input_ids_list = [feature["input_ids"] for feature in features]
        attention_mask_list = [feature["attention_mask"] for feature in features]

        # Extracts the visual features safely
        pixel_values = [feature["pixel_values"] for feature in features]
        image_grid_thw = [feature["image_grid_thw"] for feature in features]

        # 3. Dynamic Padding
        # Pad your text tensors so they are all uniform length across this batch
        padded_inputs = self.processor.tokenizer.pad(
            {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
            padding=True,
            return_tensors="pt"
        )

        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]

        # 4. Label Masking Logic
        # Create a copy of input_ids to act as your target labels
        labels = input_ids.clone()

        for i in range(len(features)):
            # Hint 3: Locate where the assistant's real response starts in this sequence
            # Turn everything BEFORE that point into -100 so the model doesn't compute loss on it
            ...

        # 5. Collate everything into a dictionary matching the model forward pass signatures
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            # Hint 4: Vision elements must be stacked or concatenated depending on processor shape
            "pixel_values": ...,
            "image_grid_thw": ...
        }

        return batch
