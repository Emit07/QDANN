#!/bin/bash
# The MPC reads on Sherlock, one array task per Iowa county, all years and the target pixels.
# A task resumes by skipping years already on disk, so a timeout or preemption costs at
# most the year in flight: resubmit the same array and only the missing work reruns.
#
# Once, on a login node (compute jobs must not download or sync, 99 at once would race):
# Sherlock's glibc (2.17) is older than the torch/pyproj/rasterio wheels need (2.28), so the
# venv is built and run inside a Debian container that has uv and Python 3.14:
#   apptainer pull py.sif docker://ghcr.io/astral-sh/uv:python3.14-bookworm
#   apptainer exec py.sif uv sync --frozen
#   apptainer exec py.sif uv run --no-sync python -c "from qdann import mpc; print(len(mpc.state_counties('19')))"  # TIGER
#   scp data/gridmet_target_maize.parquet from the laptop into data/   # --target's points
#
#   mkdir -p logs && jid=$(sbatch --parsable sherlock.sbatch)
#   sbatch --dependency=afterok:$jid sherlock_fit.sbatch   # the fit, once every task succeeds
#
#SBATCH --job-name=qdann-mpc
#SBATCH --partition=normal,owners
#SBATCH --array=0-98%25
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=16:00:00
#SBATCH --output=logs/mpc_%a.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="apptainer exec py.sif .venv/bin/python"
fips=$($python -c "from qdann import mpc; print(sorted(mpc.state_counties('19'))[$SLURM_ARRAY_TASK_ID])")
exec $python -m qdann.mpc --crop maize --years 2008-2018 --county "$fips" --target \
    --out "data/mpc/19/$fips"
