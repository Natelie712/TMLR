#!/bin/bash
#SBATCH --job-name=lambda-audit-test
#SBATCH --cpus-per-task=2
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --export=NONE
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/lambda-audit-${SLURM_JOB_ID}"
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
DIR=experiments/small_lambda_validation_test_20261110
OUT=results/small_lambda_validation_test_20261110
exec 9>"$OUT/run.lock"
flock -n 9 || exit 75
trap 'status=$?; printf "%s\n" "$status" > "$OUT/exit_${SLURM_JOB_ID}.txt"' EXIT
for model in qwen3 llama3b; do
    "$PY" -u "$DIR/gpu.py" --model "$model"
    "$PY" -u "$DIR/finish.py"
done
