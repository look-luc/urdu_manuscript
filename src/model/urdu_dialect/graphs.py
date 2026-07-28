import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch

BASE_DIR = Path(__file__).resolve().parent.parent.parent.parent

def _build_df(json_path: Path):
    train_loss = {}
    grad_norm = {}
    eval_loss = {}
    eval_bleu = {}
    eval_cer = {}
    eval_wer = {}

    temp_df = pd.read_json(json_path)

    if "log_history" in temp_df.columns:
        log_entries = temp_df["log_history"].dropna().values
    else:
        log_entries = []

    for element in log_entries:
        if not isinstance(element, dict):
            continue

        if "loss" in element:
            train_loss["step"] = element["loss"]
        if "grad_norm" in element:
            grad_norm["step"] = element["grad_norm"]
        if "eval_loss" in element:
            eval_loss["step"] = element["eval_loss"]
        if "eval_BLEU" in element:
            eval_bleu["step"] = element["eval_BLEU"]
        if "eval_CER" in element:
            eval_cer["step"] = element["eval_CER"]
        if "eval_WER" in element:
            eval_wer["step"] = element["eval_WER"]

    return train_loss, grad_norm, eval_loss, eval_bleu, eval_cer, eval_wer

def _make_graph(metric: dict[str, float], metric_name: str, color: str, output_dir: Path):
    if not metric:
        print(f"Skipping graph for {metric_name}: No log data found.")
        return

    x = torch.tensor(list(metric.keys())).numpy()
    y = torch.tensor(list(metric.values())).numpy()

    fig, ax = plt.subplots()
    ax.plot(x, y, marker='o', color=color, label=metric_name.capitalize())

    ax.set_title(f'{metric_name.capitalize()} Over Steps')
    ax.set_xlabel('Steps')
    ax.set_ylabel(metric_name.capitalize())

    ax.legend()
    ax.grid(True)

    file_name = metric_name.replace(" ", "_")
    save_path = output_dir / f"{file_name}.png"
    fig.savefig(save_path)
    plt.close(fig)

    print(f"Made graph of {metric_name} as {file_name}.png at {output_dir}")

def metrics_graph(path_to_results: str = str(BASE_DIR / "results" / "log")):
    results_dir = Path(path_to_results)
    graphs_dir = results_dir / "graphs"
    graphs_dir.mkdir(parents=True, exist_ok=True)

    json_files = list(results_dir.rglob("*.json"))
    if not json_files:
        raise FileNotFoundError(f"No .json log files found under directory: {results_dir}")

    result_json_path = json_files[0]

    metrics = _build_df(result_json_path)
    metric_names = [
        "training loss", "gradient normalization", "evaluation loss",
        "BLEU score", "CER score", "WER score"
    ]
    colors = ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#D55E00', '#CC79A7']

    for metric, name, color in zip(metrics, metric_names, colors):
        _make_graph(metric, name, color, graphs_dir)

    print("Finished graphing metrics.")
