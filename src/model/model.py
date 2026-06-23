import torch
from torchvision import io
from transformers import AutoProcessor, Qwen2VLForConditionalGeneration


class text_excraction():
    def __init__(
        self,
        model_id:str="oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct",
        prompt:str="""
            Extract all the Urdu text from this manuscript image accurately line by line. Do not revise any suffix or prefix, keep the text original as it is in the manuscript
        """,
    ) -> None:
        self.model_id = model_id
        self.prompt = prompt

    def extract(self, image:str):
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        self.processor = AutoProcessor.from_pretrained(self.model_id)
