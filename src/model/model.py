import torch
from qwen_vl_utils import process_vision_info
from torchvision import io
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)


class text_extraction:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2-VL-7B-Instruct",
        prompt: str = """
            You are a literal, character-by-character OCR extraction tool.
            Your sole task is to extract the visible text characters from the provided image.

            RULES:
            1. Output ONLY the raw extracted text wrapped inside <text> and </text> tags.
            2. extract exactly what is written, line-by-line.
            3. Stop generating immediately when you reach the blank margins or the end of the written page text.

            DIACRITIC & LIGATURE PRECISION RULES:
            1. Nastaliq script stacks words vertically. Separate vertically stacked character clusters into their distinct, individual words horizontally rather than merging them into single invented words.
            2. Pay strict attention to the placement and count of dots (nuktas). Do not substitute common modern names or words if the literal dot patterns match older historical terms.
            3. The block of text in the middle of Page 11 contains a Persian poetic couplet; extract the literal letters of these poetic rows exactly as penned without forcing standard Urdu grammar patterns.
            4. The symbol at the beginning of the bottom lines is the archaic abbreviation for 'نسخہ' (Nuskhah) — extract it accurately.
            """,
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        # Remove bnb_config entirely
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def extract(self, image_path: str, decoding_strategy: str = "greedy"):
        pytorch_image_read = io.read_image(image_path)
        image_tensor = to_pil_image(pytorch_image_read).convert("RGB")

        max_tokens = 2000

        message = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "image": image_tensor,
                        "min_pixels": 512 * 512,
                        "max_pixels": 14 * 14 * 1024 * 1024
                    },
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]

        text = self.processor.apply_chat_template(message, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(message)

        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        if decoding_strategy == "beam_search":
            gen_config = {
                "do_sample": False,
                "num_beams": 3,
                "repetition_penalty": 1.25,
                "no_repeat_ngram_size": 4,
            }
        else:
            gen_config = {
                    "do_sample": False,
                    "repetition_penalty": 1.15,       # Slightly above 1.1 to ward off the "ہور" loop
                    "no_repeat_ngram_size": 6,        # Raised to 6 so common words/letters can repeat naturally
                }

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                **gen_config
            )

        gen_id_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(gen_id_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        final_text = output_text[0].split("[END]")[0]
        return final_text
