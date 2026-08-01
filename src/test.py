import sys
from pathlib import Path

import torch
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

# Align path imports
root_dir = Path(__file__).resolve().parent
if str(root_dir) not in sys.path:
    sys.path.append(str(root_dir))

from data.get_data import get_datasets
from model.urdu_dialect.data_collector import QwenDataCollator


def run_diagnostics():
    prompt = """
        You are an expert multilingual OCR system specializing in high-accuracy transcription of Arabic, Urdu, and Persian text.
        Analyze the image carefully and transcribe the text line-by-line.
    """
    model_id = "Qwen/Qwen2.5-VL-7B-Instruct"

    # --- STAGE 1: Dataset Fetch ---
    print("\n[STAGE 1] Testing dataset streaming & image URL fetching...", flush=True)
    try:
        data = get_datasets()
        train_iter = iter(data["train"])
        sample = next(train_iter)
        print(f"-> [STAGE 1 SUCCESS] Fetched sample keys: {list(sample.keys())}", flush=True)
    except Exception as e:
        print(f"-> [STAGE 1 FAILED] Dataset stream error: {e}", flush=True)
        return

    # --- STAGE 2: Data Collator ---
    print("\n[STAGE 2] Testing QwenDataCollator processing...", flush=True)
    try:
        processor = AutoProcessor.from_pretrained(
            model_id,
            min_pixels=28 * 28,
            max_pixels=512 * 28 * 28,
        )
        collator = QwenDataCollator(processor=processor, prompt=prompt)
        batch = collator([sample])
        print(f"-> [STAGE 2 SUCCESS] Input IDs shape: {batch['input_ids'].shape}", flush=True)
    except Exception as e:
        print(f"-> [STAGE 2 FAILED] Collator error: {e}", flush=True)
        return

    # --- STAGE 3: CUDA Model Forward Pass ---
    print("\n[STAGE 3] Loading model & testing CUDA forward pass...", flush=True)
    try:
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_id,
            device_map="cuda",
            attn_implementation="sdpa",
        )

        # Move batch tensors to CUDA
        cuda_batch = {
            k: v.to("cuda") if isinstance(v, torch.Tensor) else v
            for k, v in batch.items()
        }

        print("-> Executing model forward pass...", flush=True)
        torch.cuda.synchronize()
        with torch.no_grad():
            outputs = model(**cuda_batch)
        torch.cuda.synchronize()
        print(f"-> [STAGE 3 SUCCESS] Forward pass complete! Loss: {outputs.loss.item()}", flush=True)

    except Exception as e:
        print(f"-> [STAGE 3 FAILED] GPU forward pass error: {e}", flush=True)
