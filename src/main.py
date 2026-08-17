import os
from pathlib import Path

from dotenv import load_dotenv
from huggingface_hub import login

from src.model.text_extraction.model import text_extraction
from src.model.urdu_dialect.graphs import metrics_graph
from src.model.urdu_dialect.model import unification_urdu_lang_model
from src.test import run_diagnostics
from src.uploading_model import upload_model

load_dotenv()
hf_token = os.getenv("HF_TOKEN")

if hf_token:
    login(token=hf_token)

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_PATH = SCRIPT_DIR.parent / "data"
SCRATCH_BASE = Path(f"/scratch/alpine/{os.getenv('USER', 'lude4390')}")


def run_model(what_model: str):
    if what_model == "text_extraction":
        pg10 = text_extraction().extract(str(DATA_PATH / "eval_data/pg10.png"))
        pg11 = text_extraction().extract(str(DATA_PATH / "eval_data/pg11.png"))

        output_dir = SCRATCH_BASE / "model" / "text_extraction_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
            file.write("Page 10\n")
            file.write(pg10)
            file.write("\n\nPage 11\n")
            file.write(pg11)
        print(f"Finished, model_out.txt is located in {output_dir}")

    elif what_model == "urdu_dialect":
        urdu_model = unification_urdu_lang_model()
        urdu_model.train()

        output_dir = SCRATCH_BASE / "model" / "urdu_dialect_output"
        output_dir.mkdir(parents=True, exist_ok=True)
        with open(output_dir / "model_out.txt", "w", encoding="utf-8") as file:
            file.write(f"Successfully trained model setup: {urdu_model.model_id}")

    elif what_model == "graph":
        metrics_graph()
    elif what_model == "diagnostic":
        run_diagnostics()
    elif what_model == "huggingface":
        upload_model()
    else:
        raise ValueError(f"{what_model} is not a valid model run type.")
