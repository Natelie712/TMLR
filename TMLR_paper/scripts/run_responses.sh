#!/bin/bash
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export PYTHONPATH="$PWD" HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export MPLCONFIGDIR="/tmp/tmlr-response-${SLURM_JOB_ID:-local}"
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
for model in "$@"; do
 exec 9>"TMLR_paper/data/execution/${model}.lock"
 flock -n 9 || exit 75
 "$PY" -u TMLR_paper/scripts/response_gpu.py --model "$model"
 flock -u 9
 "$PY" -u TMLR_paper/scripts/render.py
 "$PY" -u TMLR_paper/scripts/audit_figures.py
 "$PY" -u TMLR_paper/scripts/package_artifacts.py
 "$PY" -u TMLR_paper/scripts/verify_artifacts.py
 done
