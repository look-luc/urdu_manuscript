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

export SCRATCH_DIR="/scratch/alpine/$USER"
export HF_HOME="$SCRATCH_DIR/.cache/huggingface"
export EVALUATE_CACHE_DIR="$SCRATCH_DIR/.cache/evaluate"
export TRANSFORMERS_CACHE="$SCRATCH_DIR/.cache/transformers"
export TMPDIR="$SCRATCH_DIR/tmp"
export CUDA_CACHE_PATH="$TMPDIR/nv_cache"

mkdir -p "$HF_HOME" "$EVALUATE_CACHE_DIR" "$TRANSFORMERS_CACHE" "$TMPDIR" "$CUDA_CACHE_PATH"

module purge
module  load cuda
module load anaconda
conda activate urdu_manuscript_stable

SITE_PKG=$(python -c "import site; print(site.getsitepackages()[0])")
export LD_LIBRARY_PATH="$SITE_PKG/nvidia/cudnn/lib:$SITE_PKG/nvidia/cuda_runtime/lib:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH"

cd /projects/$USER/urdu_manuscript

python -c "import torch; print('CUDA Available:', torch.cuda.is_available()); print('cuDNN Enabled:', torch.backends.cudnn.enabled); print('Device Name:', torch.cuda.get_device_name(0)); x = torch.randn(2, 2).cuda(); print('Tensor CUDA Test Success!')"

MODEL_TYPE=${1:-text_extraction}
python -u run.py -o "$MODEL_TYPE"
