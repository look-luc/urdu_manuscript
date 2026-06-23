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
        self.device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_built() else "cpu"

    def setup(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id, torch_dtype=torch.float16, device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(self.model_id)

        return model, processor

    def extract(self, image_path:str):
        self.image_path = image_path
        image_tensor = io.read_image(self.image_path)

        self.model, self.processor = self.setup()

        self.message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_tensor},
                    {"type": "text", "text": self.prompt}
                ]
            }
        ]

        self.text = self.processor.apply_chat_template(self.message, tokenize=False, add_generation_prompt=True)

        self.inputs = self.processor(text=[self.text], images=image_tensor, padding=True, return_tensors="pt").to(self.device)
