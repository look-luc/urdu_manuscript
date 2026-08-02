import torch


class Data_Collector:
    def __init__(self, processor):
        self.processor = processor
        self.pad_token_id = self.processor.tokenizer.pad_token_id

        # Tokenize the assistant header delimiter
        assistant_tokens = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_start_tensor = torch.tensor(assistant_tokens, dtype=torch.long)

    def _find_subsequence(self, sequence: torch.Tensor, pattern: torch.Tensor) -> int:
        """Efficiently finds starting index of a 1D target tensor pattern in a 1D sequence tensor."""
        seq_len = sequence.size(0)
        pat_len = pattern.size(0)

        if pat_len > seq_len:
            return -1

        # Sliding window view over the sequence
        windows = sequence.unfold(0, pat_len, 1)
        matches = (windows == pattern).all(dim=1)
        indices = torch.nonzero(matches, as_tuple=True)[0]

        if len(indices) > 0:
            return indices[0].item()
        return -1

    def __call__(self, features):
        input_ids_list = [feature["input_ids"] for feature in features]
        attention_mask_list = [feature["attention_mask"] for feature in features]

        pixel_values = [feature["pixel_values"] for feature in features]
        image_grid_thw = [feature["image_grid_thw"] for feature in features]

        # Dynamic Padding for text sequences
        padded_inputs = self.processor.tokenizer.pad(
            {
                "input_ids": input_ids_list,
                "attention_mask": attention_mask_list,
            },
            padding=True,
            return_tensors="pt",
        )

        input_ids = padded_inputs["input_ids"]
        attention_mask = padded_inputs["attention_mask"]

        labels = input_ids.clone()
        pattern_len = self.assistant_start_tensor.size(0)

        # Mask prompt tokens row by row
        for i in range(len(features)):
            row_labels = labels[i]
            match_idx = self._find_subsequence(row_labels, self.assistant_start_tensor)

            if match_idx != -1:
                # Mask up to and including the assistant start marker
                mask_end = match_idx + pattern_len
                labels[i, :mask_end] = -100
            else:
                # Fallback: if header is missing, mask the full sequence to avoid prompt leakage
                labels[i, :] = -100

        # CRITICAL: Mask all padding tokens so loss isn't computed on pad_token_id
        labels[labels == self.pad_token_id] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
            "pixel_values": torch.cat(pixel_values, dim=0),
            "image_grid_thw": torch.cat(image_grid_thw, dim=0),
        }
