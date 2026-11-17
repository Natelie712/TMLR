#!/bin/bash
#SBATCH --job-name=tmlr-negative-allocated
#SBATCH --partition=notchpeak-shared-short
#SBATCH --account=notchpeak-shared-short
#SBATCH --cpus-per-task=1
#SBATCH --mem=512M
#SBATCH --time=08:00:00
#SBATCH --export=NONE
set -euo pipefail
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
exec reproductions/remy_native_linear_20261012/venv/bin/python -u TMLR_paper/scripts/allocated_negative_controller.py
