from pathlib import Path

import torch
import torchvision.io as tv_io
from peft import PeftModel
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)

script_path = root_dir = Path(__file__).resolve().parent
class text_extraction:
    def __init__(
        self,
        model_id: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        prompt: str = """
            You are a automated OCR engine operating under strict structural constraints.
            Extract the historical Urdu Nastaliq script exactly as it appears in the image.
            """,
            path_to_model:str=f"{script_path}/urdu_model"
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.path_to_model = path_to_model
        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(model, self.path_to_model)
        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def extract(self, image_path: str):
        self.model.eval()
        image_tensor = tv_io.read_image(
            image_path, mode=tv_io.ImageReadMode.RGB
        )
        image_tensor = to_pil_image(image_tensor)

        inputs = self.processor(
            image=image_tensor,
            text=self.prompt,
            return_tensors = "pt"
        ).to(self.device)

        with torch.no_grad():
            ids = self.model.generate(**inputs, max_new_tokens=2000)

        generated_ids_trimmed = [
            out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, ids)
        ]

        decoded_output = self.processor.batch_decode(
            generated_ids_trimmed,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )

        return decoded_output[0]
