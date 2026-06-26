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
module load cuda/12.1.1
module load anaconda

if [ -n "$CUDA_HOME" ]; then
    export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$CUDA_HOME/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
elif [ -n "$CUDA_PATH" ]; then
    export LD_LIBRARY_PATH="$CUDA_PATH/lib64:$CUDA_PATH/extras/CUPTI/lib64:$LD_LIBRARY_PATH"
else
    echo "WARNING: CUDA_HOME is not set. CUPTI path injection may fail."
fi

set +u && conda activate urdu_manuscript_stable && set -u

cd /projects/$USER/urdu_manuscript

python -u run.py -o "text_extraction"
