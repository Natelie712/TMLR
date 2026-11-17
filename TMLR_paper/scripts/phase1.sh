#!/bin/bash
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/tmlr-paper-mpl
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
exec 8>TMLR_paper/data/execution/cpu.lock
flock -n 8 || exit 75
trap 'printf "%s\n" "$?" > TMLR_paper/data/execution/cpu_exit_code.txt' EXIT
"$PY" -u TMLR_paper/scripts/inventory.py
"$PY" -u TMLR_paper/scripts/build_tables.py
"$PY" -u TMLR_paper/scripts/geometry.py
