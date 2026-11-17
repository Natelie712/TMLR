#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")/.."
export MPLBACKEND=Agg OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2
PY=${PYTHON:-python}
"$PY" scripts/verify_release.py
"$PY" scripts/render_release.py
