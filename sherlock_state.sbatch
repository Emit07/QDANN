#!/bin/bash
# Stage 2 (notes/yield_layer_plan.md): MPC county means for every county of KS/MO/SD,
# 2008-2023, one array task per county, county means only (no target pixels). Same
# container, venv and resume rule as sherlock_extend.sbatch: a task skips years on disk,
# so resubmitting the same array reruns only what is missing.
#
# Landsat 7 out after 2021 (AMBIGUITIES #30), as in the reads behind
# source_mpc_maize_2008_2023_l7_2021.parquet: 2008-2021 read as the 2008-2018 Iowa reads did.
# The array runs over every county of the three states in sorted order, so %25 caps the MPC
# readers at 25 in all, as for Iowa.
#
# Once, on a login node, after pulling the commit that adds this file:
#   apptainer exec py.sif uv sync --frozen
#   # CropScape's CDL for 2022-2023 (mpc.MPC_CDL_LAST), fetched once here rather than by
#   # every task at once:
#   apptainer exec py.sif .venv/bin/python -c "from qdann import mpc; [mpc.cropscape(y, s) for y in (2022, 2023) for s in ('20', '29', '46')]"
#   n=$(apptainer exec py.sif .venv/bin/python -c "from qdann import mpc; print(sum(len(mpc.state_counties(s)) for s in ('20', '29', '46')))")
#   mkdir -p logs && sbatch --array=0-$((n - 1))%25 sherlock_state.sbatch
#
# Afterwards: grep logs/state_*.out for "skipping unreadable scene", copy
# $SCRATCH/qdann/mpc_l7_2021/{20,29,46} to the laptop's data/mpc_l7_2021/, and run
# data/stage2/build.py there (fit, completeness check, join).
#
#SBATCH --job-name=qdann-state
#SBATCH --partition=normal,owners
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=24:00:00
#SBATCH --output=logs/state_%a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="apptainer exec py.sif .venv/bin/python"
fips=$($python -c "
from qdann import mpc
print(sorted(f for s in ('20', '29', '46') for f in mpc.state_counties(s))[$SLURM_ARRAY_TASK_ID])")
out="$SCRATCH/qdann/mpc_l7_2021/${fips:0:2}/$fips"
exec $python -m qdann.mpc --crop maize --years 2008-2023 --county "$fips" --out "$out" \
    --landsat7-last 2021
