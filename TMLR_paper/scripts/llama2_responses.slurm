#!/bin/bash
#SBATCH --job-name=tmlr-llama2-response
#SBATCH --cpus-per-task=2
#SBATCH --mem=64G
#SBATCH --time=06:00:00
#SBATCH --export=NONE
set -euo pipefail
export KPCA_RESULTS_DIR=/scratch/rai/vast1/u1557683/project_storage/kpca_steering/results
cd /uufs/chpc.utah.edu/common/home/u1557683/projectscode/kpca_steering
bash TMLR_paper/scripts/run_responses.sh llama2
