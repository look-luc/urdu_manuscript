import os
from pathlib import Path
import torch

import matplotlib.pyplot as plt
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent

def _build_df(json:str):
    train_loss = {}
    grad_norm = {}
    eval_loss = {}
    eval_bleu = {}
    eval_cer = {}
    eval_wer = {}

    temp_df = pd.read_json(json, lines=True)

    for element in temp_df["log_history"].values():
        if "loss" in element.keys():
            train_loss[element["step"]] = element["loss"]
            grad_norm[element["step"]] = element["grad_norm"]
        if "eval_loss" in element.keys():
            eval_loss[element["step"]] = element["eval_loss"]
            eval_bleu[element["step"]] = element["eval_BLEU"]
            eval_cer[element["step"]] = element["eval_CER"]
            eval_wer[element["step"]] = element["eval_WER"]
    return train_loss, grad_norm, eval_loss, eval_bleu, eval_cer, eval_wer

def _make_graph(metric:dict[str:float], metric_name:str, color:str, path=""):
    x = torch.tensor(list(metric.keys())).numpy()
    y = torch.tensor(list(metric.values())).numpy()
    plt.plot(x, y, marker='o', color=color, label=metric_name.capitalize())

    plt.title(f'{metric_name.capitalize()} Over Epochs')

    plt.xlabel('Epochs')
    plt.ylabel(metric_name.capitalize())

    plt.legend()
    plt.grid(True)

    file_name = metric_name.replace(" ", "_")
    plt.savefig(f"{BASE_DIR}/results/graphs/{file_name}.png")

    print(f"made graph of {metric_name} as {file_name}.png at {BASE_DIR}/results/graphs")

def metrics_graph(path_to_results:str=f"{BASE_DIR}/results/log"):
    path = Path(f"{path_to_results}/graphs")
    path.mkdir(parents=True, exist_ok=True)

    result_jsonl = str(path_to_results.rglob("*.jsonl"))

    train_loss, grad_norm, eval_loss, eval_bleu, eval_cer, eval_wer = _build_df(result_jsonl)

    metrics = [train_loss, grad_norm, eval_loss, eval_bleu, eval_cer, eval_wer]
    metric_names = ["training loss", "gradient normalization", "evaluation loss", "BLEU score", "CER score", "WER score"]
    colors = ['#0072B2', '#E69F00', '#009E73', '#F0E442', '#D55E00', '#CC79A7']
    for metric, name, color in zip(metrics, metric_names, colors):
