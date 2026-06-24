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

export HF_HOME="/projects/$USER/.cache/huggingface"
mkdir -p "$HF_HOME"

module purge
module load cuda/12.x
module load anaconda
set +u && conda activate urdu_manuscript_stable && set -u

cd /projects/$USER/urdu_manuscript/src

python -u main.py
