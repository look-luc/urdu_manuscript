import torch
import torchvision.io as tv_io


class Data_Collector:
    def __init__(self, processor, prompt: str = ""):
        self.processor = processor
        self.prompt = prompt
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
        image_tensors = []
        text_prompts = []

        for feature in features:
            raw_bytes = feature["image_bytes"]

            # Decode binary image byte buffer into a 3D PyTorch RGB Tensor (C, H, W)
            byte_tensor = torch.frombuffer(bytearray(raw_bytes), dtype=torch.uint8)
            img_tensor = tv_io.decode_image(byte_tensor, mode=tv_io.ImageReadMode.RGB)
            image_tensors.append(img_tensor)

            txt_content = feature.get("text", "")

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image", "image": img_tensor},
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

        # Process PyTorch tensors directly via Qwen AutoProcessor
        batch = self.processor(
            text=text_prompts,
            images=image_tensors,
            padding=True,
            return_tensors="pt",
        )

        input_ids = batch["input_ids"]
        labels = input_ids.clone()
        pattern_len = self.assistant_start_tensor.size(0)

        # Mask user prompt tokens row by row
        for i in range(len(features)):
            row_labels = labels[i]
            match_idx = self._find_subsequence(row_labels, self.assistant_start_tensor)

            if match_idx != -1:
                # Mask up to and including the assistant start marker
                mask_end = match_idx + pattern_len
                labels[i, :mask_end] = -100
            else:
                # Fallback: mask full sequence if header is missing
                labels[i, :] = -100

        # Mask padding tokens so loss isn't computed on pad_token_id
        labels[labels == self.pad_token_id] = -100
        batch["labels"] = labels

        return batch
