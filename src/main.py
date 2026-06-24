import os

from dotenv import load_dotenv

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

from pathlib import Path

import torch

from model.model import text_extraction

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR.parent / "data"

def main():
    image_path = "../data/Original_manuscript_old_urdu.jpeg"
    print(f"Checking for file at: {os.path.abspath(image_path)}")
    print(f"Does file exist? {os.path.exists(image_path)}")

    model = text_extraction().extract(str(DATA_PATH/"Original_manuscript_old_urdu.jpeg"))

    with open("model_out.txt", "w", encoding="utf-8") as file:
        file.write(model)
    return 0

if __name__ == "__main__":
    main()
