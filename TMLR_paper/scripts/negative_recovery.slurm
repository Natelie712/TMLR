#!/bin/bash
#SBATCH --job-name=tmlr-negative-recovery
#SBATCH --partition=notchpeak-shared-short
#SBATCH --account=notchpeak-shared-short
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=08:00:00
#SBATCH --export=NONE
set -euo pipefail
export KPCA_RESULTS_DIR=/scratch/rai/vast1/u1557683/project_storage/kpca_steering/results
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
exec reproductions/remy_native_linear_20261012/venv/bin/python -u TMLR_paper/scripts/negative_recovery.py
