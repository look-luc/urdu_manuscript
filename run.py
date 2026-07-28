import argparse
import gc

import torch

from src.main import run_model


def run(model_type:str):
    pass

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()

    parser = argparse.ArgumentParser(description="all arguments to see what type of model will run")

    parser.add_argument("-m", "--model-type", type=str, help="Insert what model type want to use (text_extraction, urdu_dialect, or graph)")

    args = parser.parse_args()

    print(f"running {args.model_type}\n\n")
    run_model(args.model_type)
