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
        prompt: str = """You are an automated OCR engine operating under strict structural constraints.
                Extract the historical Urdu Nastaliq text exactly as it is written in the image.

                CRITICAL OUTPUT FORMATTING RULES:
                1. Output ONLY the raw extracted text wrapped inside <text> and </text> tags.
                2. Do NOT write any introductory or concluding remarks.
                3. Do NOT wrap the output in markdown code blocks (```).
                4. Maintain exact line-by-line formatting matching the rows of the manuscript.

                VISUAL RENDERING RULES:
                1. Extract the literal Urdu letters you see. Do NOT attempt to convert the text into classical Arabic or add full Quranic vowel diacritics.
                2. Only Extract vowel markings (اعراب) if they are explicitly, darkly drawn on the page.
                3. Retain archaic Dakhni vocabulary elements (e.g., 'کوں', 'ہور', 'سون') exactly as written.
                4. If a line ends or a new visual row begins, immediately insert a newline formatting break.""",
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
