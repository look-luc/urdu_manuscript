import torch
from torchvision import io
from torchvision.transforms.functional import to_pil_image
from transformers import TrOCRProcessor, VisionEncoderDecoderModel


class text_extraction:
    def __init__(
        self,
        model_id: str = "microsoft/trocr-base-handwritten",
    ) -> None:
        torch.backends.cudnn.enabled = False

        self.model_id = str(model_id)
        self.prompt = prompt
        self.device = str("cuda") if torch.cuda.is_available() else "cpu"

        self.model, self.processor = self._setup_model()

    def _setup_model(self):
        model = VisionEncoderDecoderModel.from_pretrained(self.model_id).to(self.device)
        processor = TrOCRProcessor.from_pretrained(self.model_id)

        return model, processor

    def extract(self, image_path: str):
        pytorch_image_read = io.read_image(image_path)
        image_tensor = to_pil_image(pytorch_image_read).convert("RGB")

        pixel_values = self.processor(images=image, return_tensors="pt").pixel_values.to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(pixel_values)

        output_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return final_text
