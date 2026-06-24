import argparse
from pathlib import Path

from dotenv import load_dotenv
from model.model.text_extraction import text_extraction
from model.model.urdu_dialect import unification_urdu_lang_model

load_dotenv()
# os.environ["HF_TOKEN"] = os.getenv("HUGGING_FACE_TOKEN")

SCRIPT_DIR = Path(__file__).resolve().parent

DATA_PATH = SCRIPT_DIR.parent / "data"

def run_model(what_model:str):
    if what_model == "text_extraction":
        right_side_model = text_extraction().extract(str(DATA_PATH/"page_10_original_manuscript_old_urdu.jpeg"), "greedy")
        left_side_model = text_extraction().extract(str(DATA_PATH/"page_11_original_manuscript_old_urdu.jpg"), "greedy")

        output_dir = Path("./model/text_extraction")
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
            file.write("Page 10 (right page)\n")
            file.write(right_side_model)
            file.write("\n\nPage 11 (left page)\n")
            file.write(left_side_model)
        print(f"Finished, model_out.txt is located in {output_dir}")
    elif what_model == "urdu_dialect":
        pass
    else:
        raise ValueError(f"{what_model} is not a valid model run type.")
