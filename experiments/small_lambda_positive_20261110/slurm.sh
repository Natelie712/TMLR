#!/bin/bash
#SBATCH --job-name=small-lambda-positive
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/small-lambda-${SLURM_JOB_ID}"
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
DIR=experiments/small_lambda_positive_20261110
OUT=results/small_lambda_positive_20261110
exec 9>"$OUT/run.lock"
flock -n 9 || exit 75
trap 'status=$?; printf "%s\n" "$status" > "$OUT/slurm_exit_${SLURM_JOB_ID}.txt"' EXIT
"$PY" -u -c 'import torch; assert torch.cuda.is_available(); print(torch.cuda.get_device_name(0), torch.ones(1,device="cuda"),flush=True)'
for partition in validation fit; do
    for model in qwen3 llama3b; do
        "$PY" -u "$DIR/execute.py" --model "$model" --partition "$partition"
        "$PY" -u "$DIR/report.py"
    done
done
