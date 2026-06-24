import os

from dotenv import load_dotenv

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

from model.model import text_extraction


def main():
    model = text_extraction().extract("../data/Original_manuscript_old_urdu.jpeg")

    with open("model_out.txt", "w", encoding="utf-8") as file:
        file.write(model)
    return 0

if __name__ == "__main__":
    main()
