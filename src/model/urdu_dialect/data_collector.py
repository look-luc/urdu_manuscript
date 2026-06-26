import torch


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

        # Dynamic Padding
        # Pad the text tensors so they are all uniform length across this batch
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list
            },
            padding=True,
            return_tensors="pt"
        )

        input_ids = padded_inputs["input_ids"] # the ids for the inputs after being padded
        attention_mask = padded_inputs["attention_mask"] # attention masks after being padded

        # Create a copy of input_ids to act as your target labels
        labels = input_ids.clone()

        # going row by row
        for i in range(len(features)):
            row_labels = labels[i]

            for row in range(len(row_labels)-len(self.assistant_start_token)+1):
                # determining if window of size current row to the current row + how ever long the assistant start token is
                if row_labels[row:row+len(self.assistant_start_token)].tolist() == self.assistant_start_token:
                    labels[i, :row+len(self.assistant_start_token)] = -100 #will assign the range to the value specified; labels is a torch tensor not a list
                    break

        # Collate everything into a dictionary matching the model forward pass signatures
        batch = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw, dim=0)
        }

        return batch
