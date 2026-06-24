from pathlib import Path

from dotenv import load_dotenv

from model.model import text_extraction

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR.parent / "data"

def main():
    right_side_model = text_extraction().extract(str(DATA_PATH/"page_10_original_manuscript_old_urdu.jpeg"), "beam_search")
    left_side_model = text_extraction().extract(str(DATA_PATH/"page_11_original_manuscript_old_urdu.jpeg"), "beam_search")

    with open("model_out.txt", "w", encoding="utf-8") as file:
        file.write("Page 10 (right page)\n")
        file.write(right_side_model)
        file.write("\n\nPage 11 (left page)\n")
        file.write(left_side_model)
    return 0

if __name__ == "__main__":
    main()
