#!/bin/bash
# CPU artifacts by default; GPU diagnostic execution requires an explicit argument.
set -euo pipefail
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 MKL_NUM_THREADS=2 MPLBACKEND=Agg
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 MPLCONFIGDIR=/tmp/tmlr-paper-reproduce
PY="$PWD/reproductions/remy_native_linear_20261012/venv/bin/python"
case "${1:-cpu}" in
 cpu)
  "$PY" TMLR_paper/scripts/inventory.py
  "$PY" TMLR_paper/scripts/build_tables.py
  if [ ! -f TMLR_paper/data/geometric_diagnostics/completion.json ]; then "$PY" TMLR_paper/scripts/geometry.py; fi
  if [ ! -f TMLR_paper/data/geometric_diagnostics/fit_correlation_completion.json ]; then "$PY" TMLR_paper/scripts/fit_correlations.py; fi
  "$PY" TMLR_paper/scripts/extra_audit.py
  "$PY" TMLR_paper/scripts/render.py
  "$PY" TMLR_paper/scripts/audit_figures.py
  "$PY" TMLR_paper/scripts/package_artifacts.py
  "$PY" TMLR_paper/scripts/verify_artifacts.py
  ;;
 freeze) "$PY" TMLR_paper/scripts/freeze.py ;;
 gpu)
  shift
  bash TMLR_paper/scripts/run_responses.sh "$@"
  ;;
 report) "$PY" TMLR_paper/scripts/render.py; "$PY" TMLR_paper/scripts/audit_figures.py; "$PY" TMLR_paper/scripts/package_artifacts.py; "$PY" TMLR_paper/scripts/verify_artifacts.py ;;
 *) printf 'Usage: reproduce.sh [cpu|freeze|gpu MODEL...|report]\n'; exit 2 ;;
esac
