#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/projects/%u/urdu_manuscript/logs/%j.log
#SBATCH --job-name=urdu_text_extraction
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

export HF_TOKEN="${HF_TOKEN}"
export HF_HOME="/projects/$USER/.cache/huggingface"
export EVALUATE_CACHE_DIR="/projects/$USER/.cache/evaluate"
export TRANSFORMERS_CACHE="/projects/$USER/.cache/transformers"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE"

module purge
module load anaconda
conda activate urdu_manuscript_stable

cd /projects/$USER/urdu_manuscript
pip install jiwer --quiet

MODEL_TYPE=${1:-text_extraction}
python -u run.py -o "$MODEL_TYPE"
