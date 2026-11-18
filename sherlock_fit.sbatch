#!/bin/bash
# Eq. 2 over every county directory sherlock.sbatch read: writes gcvi_mpc_maize.parquet and
# gcvi_target_mpc_maize.parquet under $SCRATCH/qdann/mpc/19. Chain it after the read array so it
# starts only once every county succeeded:
#
#   jid=$(sbatch --parsable sherlock.sbatch)
#   sbatch --dependency=afterok:$jid sherlock_fit.sbatch
#
#SBATCH --job-name=qdann-fit
#SBATCH --partition=normal,owners
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=2:00:00
#SBATCH --output=logs/fit.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
exec apptainer exec py.sif .venv/bin/python -m qdann.mpc --crop maize --fit "$SCRATCH/qdann/mpc/19"
