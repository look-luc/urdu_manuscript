import os

from dotenv import load_dotenv

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

import torch

from model.model import text_extraction


def main():
    print(f"Cuda available: {torch.cuda.is_available()}")
    print(f"Cuda device count: {torch.cuda.device_count()}")
    if torch.cuda.is_available():
        print(f"Current device: {torch.cuda.current_device()}")
        print(f"Device name: {torch.cuda.get_device_name(0)}")
        print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")

    model = text_extraction().extract("../data/Original_manuscript_old_urdu.jpeg")

    with open("model_out.txt", "w", encoding="utf-8") as file:
        file.write(model)
    return 0

if __name__ == "__main__":
    main()
