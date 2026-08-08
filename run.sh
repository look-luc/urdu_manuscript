#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=6:00:00
#SBATCH --output=/projects/%u/urdu_manuscript/logs/%j.log
#SBATCH --job-name=urdu_text_extraction
#SBATCH --partition=blanca-clearlab2
#SBATCH --account=blanca-clearlab2
#SBATCH --qos=blanca-clearlab2
#SBATCH --mail-type=END,FAIL

export HF_TOKEN="${HF_TOKEN}"

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export TOKENIZERS_PARALLELISM=false
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

export CUDA_LAUNCH_BLOCKING=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

export SCRATCH_DIR="/scratch/alpine/$USER"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export EVALUATE_CACHE_DIR="$SCRATCH_DIR/.cache/evaluate"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/.cache/transformers"
export TMPDIR="$SCRATCH_DIR/tmp"
export CUDA_CACHE_PATH="$TMPDIR/nv_cache"

cleanup() {
    echo "Cleaning up temporary files in $TMPDIR..."
    rm -rf "$TMPDIR"/*
}
trap cleanup EXIT

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE" "$TMPDIR" "$CUDA_CACHE_PATH"

module purge
module load anaconda

conda activate urdu_manuscript_stable

export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

cd /projects/$USER/urdu_manuscript

MODEL_TYPE=${1:-text_extraction}
python -u run.py -o "$MODEL_TYPE"
