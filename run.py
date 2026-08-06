import argparse
import gc

import torch
import torch.multiprocessing as mp

from src.main import run_model

if __name__ == "__main__":
    gc.collect()
    torch.cuda.empty_cache()
    mp.set_start_method("spawn", force=True)

    parser = argparse.ArgumentParser(description="all arguments to see what type of model will run")

    parser.add_argument("-o", "--override", type=str, help="Insert what model type want to use (text_extraction, urdu_dialect, diagnostic, or graph)")

    args = parser.parse_args()

    print(f"running {args.override}\n\n")
    run_model(args.override)
