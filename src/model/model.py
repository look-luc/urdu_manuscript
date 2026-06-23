import torch
from qwen_vl_utils import process_vision_info
from torchvision import io
from torchvision.transforms.functional import to_pil_image
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
            self.model_id,
            torch_dtype="auto",
            device_map="auto"
        )

        processor = AutoProcessor.from_pretrained(self.model_id)

        return model, processor

    def extract(self, image_path:str):
        self.image_path = image_path
        pytorch_image_read = io.read_image(self.image_path)
        image_tensor = to_pil_image(pytorch_image_read).convert("RGB")

        self.model, self.processor = self.setup()
        max_tokens = 2000

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

        image_inputs, video_inputs = process_vision_info(self.message)

        self.inputs = self.processor(
            text=[self.text],
            images=image_inputs,
            videos=None,
            padding=True,
            return_tensors="pt"
        ).to(self.device)

        generated_ids = self.model.generate(**self.inputs, max_new_tokens=max_tokens)
        gen_id_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(self.inputs.input_ids, generated_ids)]

        output_text = self.processor.batch_decode(gen_id_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0]
