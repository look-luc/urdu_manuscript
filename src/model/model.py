import torch
from qwen_vl_utils import process_vision_info
from torchvision import io
from torchvision.transforms.functional import to_pil_image
from transformers import (
    AutoProcessor,
    BitsAndBytesConfig,
    Qwen2VLForConditionalGeneration,
)


class text_extraction:
    def __init__(
        self,
        model_id: str = "oddadmix/Qaari-0.1-Urdu-OCR-VL-2B-Instruct",
        prompt: str = """You are a scholar trying to digitize this manuscript. Put indicators where each page starts and ends to know for later transliteration. Extract the Urdu text in the image. Ignore background noise, page stains, and bleed-through. If you reach the end of the text or cannot recognize a character, output [END]. Output the raw text directly. Do not output bounding box coordinates, spatial locations, or detection data. Provide only the extracted text characters.""",
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = model_id
        self.prompt = prompt
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True
        )

        model = Qwen2VLForConditionalGeneration.from_pretrained(
            self.model_id,
            torch_dtype="auto",
            device_map={"": 0},
            trust_remote_code=True,
            quantization_config=bnb_config,
        )

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
                temperature=0.1,         # Keeps token selection tight and near-greedy
                repetition_penalty=1.1,  # Just enough penalty to nudge it out of the "اسے" loop
                top_p=0.9
            )

        gen_id_trimmed = [out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)]
        output_text = self.processor.batch_decode(gen_id_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False)

        final_text = output_text[0].split("[END]")[0]
        return final_text
