#!/bin/bash
#SBATCH --job-name=llama2-small-lambda
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --export=NONE
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export KPCA_RESULTS_DIR=/scratch/rai/vast1/u1557683/project_storage/kpca_steering/results
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/llama2-small-lambda-${SLURM_JOB_ID}"
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
DIR=experiments/llama2_small_lambda_positive_20261111
OUT="$KPCA_RESULTS_DIR/llama2_small_lambda_positive_20261111"
exec 9>"$OUT/run.lock"
flock -n 9 || exit 75
trap 'status=$?; printf "%s\n" "$status" > "$OUT/slurm_exit_${SLURM_JOB_ID}.txt"' EXIT
"$PY" -u -c 'import torch; assert torch.cuda.is_available(); assert torch.cuda.get_device_properties(0).total_memory > 40*1024**3; print(torch.cuda.get_device_name(0), torch.ones(1,device="cuda"),flush=True)'
if [ ! -f "$OUT/preparation_complete.json" ]; then
    "$PY" -u "$DIR/prepare.py"
fi
"$PY" -u "$DIR/execute.py"
