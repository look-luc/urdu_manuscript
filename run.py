import argparse

from src.main import run_model


def run(model_type:str):
    pass

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="all arguments to see what type of model will run")

    parser.add_argument("-o", "--override", type=str, help="Insert what model type want to use (text_extraction or urdu_dialect)")

    args = parser.parse_args()

    print(f"running {args.override}\n\n")
    print(run_model(args.override))
