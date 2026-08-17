import torch
from peft import PeftModel
from transformers import (
    AutoModelForImageTextToText,
    AutoModelForMultimodalLM,
    AutoProcessor,
)


def upload_model():
    torch.backends.cudnn.enabled = False
    base_model_id = "Qwen/Qwen2.5-VL-3B-Instruct"
    adapter_path = "./src/model/text_extraction/urdu_manusript_model"
    output_dir = "./urdu_manuscript"
    hub_repo_id = "lookitsluc1/urdu-manuscript-text-extraction"

    model = AutoModelForMultimodalLM.from_pretrained(
        base_model_id,
        dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )

    peft_model = PeftModel.from_pretrained(model, adapter_path)
    merged_model = peft_model.merge_and_unload()

    # 3. Configure processor and image scaling bounds
    processor = AutoProcessor.from_pretrained(base_model_id)
    processor.image_processor.min_pixels = 256 * 28 * 28
    processor.image_processor.max_pixels = 1024 * 28 * 28

    merged_model.save_pretrained(output_dir)
    processor.save_pretrained(output_dir)

    merged_model.push_to_hub(hub_repo_id)
    processor.push_to_hub(hub_repo_id)
