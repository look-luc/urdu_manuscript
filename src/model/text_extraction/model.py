from pathlib import Path

import pandas as pd
import torch
from peft import PeftModel, get_peft_model
from qwen_vl_utils import process_vision_info
from safetensors import safe_open
from torchvision import io
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
        with torch.no_grad():
