import torch
from transformers import AutoModelForVision2Seq, AutoProcessor


class unification_urdu_lang_model:
    def __init__(self)->None:
        torch.backends.cudnn.enabled = False
        self.model_id="Qwen/Qwen3.5-9B"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model, self.processor = self._setup()

    def _setup (self):
        model = AutoModelForVision2Seq.from_pretrained(
            model_id,
            torch_dtype=torch.bfloat16,
            device_map="auto"
        )
        processor = AutoProcessor.from_pretrained(model_id)

        return  model, processor
