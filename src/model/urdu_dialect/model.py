import torch
from transformers import (
    AutoProcessor,
    Qwen2VLForConditionalGeneration,
)


class unification_urdu_lang_model:
    def __init__(
        self,
        model_id:str="Qwen/Qwen2.5-VL-72B-Instruct",
        prompt:str="""
            You are a multilingual OCR model. You must extract the text given to you.
        """
    )->None:
        torch.backends.cudnn.enabled = False
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model_id=model_id
        self.model, self.processor = self._setup()

        self.prompt = prompt

    def _setup (self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
        )
        processor = AutoProcessor.from_pretrained(self.model_id)

        return  model, processor

    def train(self, data):
        self.data = data

        max_tokens = 2000

        image_tensor = None

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
