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
        model_id: str = "Qwen/Qwen2-VL-2B-Instruct",
        prompt: str = """You are a scholar trying to digitize this manuscript. Extract the Urdu text in the image. Ignore background noise, page stains, and bleed-through. Output the raw text directly. Do not output bounding box coordinates, spatial locations, or detection data. Provide only the extracted text characters.""",
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to(self.device)

        processor = AutoProcessor.from_pretrained(self.model_id)
        return model, processor

    def extract(self, image_path: str):
        pytorch_image_read = io.read_image(image_path)
        image_tensor = to_pil_image(pytorch_image_read).convert("RGB")

        max_tokens = 2000

        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image_tensor},
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

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                do_sample=True,
                temperature=0.2,
                repetition_penalty=1.2,
                top_p=0.9
            )

        gen_id_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(gen_id_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        return output_text[0]
