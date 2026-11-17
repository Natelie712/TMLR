#!/bin/bash
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
DIR=experiments/small_lambda_positive_20261110
OUT=results/small_lambda_positive_20261110
exec 9>"$OUT/run.lock"
flock -n 9 || exit 75
trap 'status=$?; printf "%s\n" "$status" > "$OUT/launcher_exit_code.txt"' EXIT
"$PY" -u "$DIR/prepare.py"
for partition in validation fit; do
    for model in qwen3 llama3b; do
        "$PY" -u "$DIR/execute.py" --model "$model" --partition "$partition"
        "$PY" -u "$DIR/report.py"
    done
done
