#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/projects/%u/urdu_manuscript/logs/%j.log
#SBATCH --job-name=urdu_text_extraction
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

rm -rf /projects/$USER/.cache/huggingface/hub/*
export HF_HOME="/projects/$USER/.cache/huggingface"
mkdir -p $HF_HOME

set -uo pipefail

REPO_ROOT="/projects/$USER"
cd "$REPO_ROOT"

cd "$REPO_ROOT/urdu_manuscript/src"

/projects/$USER/software/anaconda/envs/urdu_manuscript_stable/bin/python main.py
