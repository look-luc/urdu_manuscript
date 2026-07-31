import torch


class Data_Collector:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id
        self.assistant_start_token = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )

    def __call__(self, features):
        if not features:
            raise ValueError("Empty batch or all samples failed validation")

        features = [
            f for f in features if f is not None and f.get("is_valid", False)
        ]
        if len(features) == 0:
            raise ValueError(
                "Data_Collector received an empty batch or all samples in the batch failed validation."
            )

        input_ids_list = [feature["input_ids"] for feature in features]
        attention_mask_list = [
            feature["attention_mask"] for feature in features
        ]
        padded_text = self.processor.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            },
            padding=True,
            return_tensors="pt",
        )

        pixel_values_list = []
        image_grid_thw_list = []
        for feature in features:
            pixels = feature["pixel_values"]
            grid_thw = feature["image_grid_thw"]

            while pixels.dim() > 2:
                pixels = pixels.squeeze(0)
            if pixels.dim() == 1:
                pixels = pixels.unsqueeze(0)

            while grid_thw.dim() > 2:
                grid_thw = grid_thw.squeeze(0)
            if grid_thw.dim() == 1:
                grid_thw = grid_thw.unsqueeze(0)

            pixel_values_list.append(pixels)
            image_grid_thw_list.append(grid_thw)

        pixel_values = torch.cat(pixel_values_list, dim=0)
        image_grid_thw = torch.cat(image_grid_thw_list, dim=0)

        if image_grid_thw.dim()==3 and image_grid_thw.size(1):
            image_grid_thw = image_grid_thw.squeeze(1)

        labels = padded_text["input_ids"].clone()
        for i in range(len(features)):
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
        labels[padded_text["input_ids"] == self.pad_token_id] = -100

        return {
            "input_ids": padded_text["input_ids"],
            "attention_mask": padded_text["attention_mask"],
            "labels": labels,
            "pixel_values": pixel_values.bfloat16(),
            "image_grid_thw": image_grid_thw,
        }
