import os
from pathlib import Path

import torch
from dotenv import load_dotenv

from model.model import text_extraction

load_dotenv()

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR.parent / "data"

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version()}")
    print(f"GPU Name: {torch.cuda.get_device_name(0)}")
    print(f"GPU Capability: {torch.cuda.get_device_capability(0)}")

    model = text_extraction().extract(str(DATA_PATH/"Original_manuscript_old_urdu.jpeg"))

    with open("model_out.txt", "w", encoding="utf-8") as file:
        file.write(model)
    return 0

if __name__ == "__main__":
    main()
