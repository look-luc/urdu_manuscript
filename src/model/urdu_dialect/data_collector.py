import torch


class Data_Collector:
    def __init__(self, processor, prompt: str = ""):
        self.processor = processor
        self.prompt = prompt
        self.pad_token_id = self.processor.tokenizer.pad_token_id

        assistant_tokens = self.processor.tokenizer.encode(
            "<|im_start|>assistant\n", add_special_tokens=False
        )
        self.assistant_start_tensor = torch.tensor(
            assistant_tokens, dtype=torch.long
        )

    def _find_subsequence(
        self, sequence: torch.Tensor, pattern: torch.Tensor
    ) -> int:
        seq_len = sequence.size(0)
        pat_len = pattern.size(0)

        if pat_len > seq_len:
            return -1

        windows = sequence.unfold(0, pat_len, 1)
        matches = (windows == pattern).all(dim=1)
        indices = torch.nonzero(matches, as_tuple=True)[0]

        if len(indices) > 0:
            return indices[0].item()
        return -1

    def __call__(self, features):
        images = []
        text_prompts = []

        for feature in features:
            # Extract the PIL Image directly from the "image" key
            img = feature["image"]
            images.append(img)

            txt_content = feature.get("text", "")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img},
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

            formatted_text = self.processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=False
            )
            text_prompts.append(formatted_text)

        batch = self.processor(
            text=text_prompts,
            images=images,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pattern_len = self.assistant_start_tensor.size(0)

        for i in range(len(features)):
            row_labels = labels[i]
            match_idx = self._find_subsequence(
                row_labels, self.assistant_start_tensor
            )

            if match_idx != -1:
                labels[i, : match_idx + pattern_len] = -100
            else:
                labels[i, :] = -100

        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels

        return batch
