from pathlib import Path

import torch
import torchvision.io as tv_io
from peft import PeftModel
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoModelForImageTextToText,
    AutoProcessor,
)

script_path = root_dir = Path(__file__).resolve().parent
class text_extraction:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = """You are an expert OCR model for historical Urdu and Arabic-script manuscripts with expert knowledge of Farsi/Persian, Arabic and Urdu. Transcribe the text line-by-line. If there are marginal notes or footnotes, transcribe them separately at the end under 'Marginalia'. Do not translate.""",
            path_to_model:str=f"{script_path}/urdu_manuscript_model"
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path_to_model = path_to_model
        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        model = AutoModelForImageTextToText.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )

        peft_model = PeftModel.from_pretrained(model, self.path_to_model)
        peft_model = peft_model.merge_and_unload()

        processor = AutoProcessor.from_pretrained(self.model_id)

        processor.image_processor.min_pixels=256 * 28 * 28
        processor.image_processor.max_pixels=1024 * 28 * 28

        return peft_model, processor

    def extract(self, pth_to_img: str):
        self.model.eval()

        message_text = [
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "text", "text": self.prompt}
                    ]
                }
            ]
        text = self.processor.apply_chat_template(
            message_text, tokenize=False, add_generation_prompt=True
        )
        image_tensor = tv_io.read_image(
            pth_to_img, mode=tv_io.ImageReadMode.RGB
        )
        image_tensor = to_pil_image(image_tensor)

        inputs = self.processor(
            images=image_tensor,
            text=text,
            return_tensors = "pt"
        ).to(self.device)

        with torch.no_grad():
            ids = self.model.generate(
                **inputs,
                max_new_tokens=2000,
                repetition_penalty=1.15,
            )

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, ids)
        ]

        decoded_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
        )

        return decoded_output[0]
