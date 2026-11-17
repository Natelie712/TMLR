#!/bin/bash
#SBATCH --job-name=tmlr-negative-lambda
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
set -euo pipefail
export KPCA_RESULTS_DIR=/scratch/rai/vast1/u1557683/project_storage/kpca_steering/results
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/tmlr-negative-${SLURM_JOB_ID}"
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
for model in qwen3 llama3b llama2; do
  "$PY" -u TMLR_paper/scripts/negative_resume.py run --model "$model"
done
