#!/bin/bash
# Stage 2 (notes/yield_layer_plan.md): MPC county means for KS/MO/SD, 2008-2023, only the
# county-years with a NASS label (data/stage2/reads.txt, written by data/stage2/build.py),
# one array task per labelled county, county means only (no target pixels). Same
# container, venv and resume rule as sherlock_extend.sbatch: a task skips years on disk,
# so resubmitting the same array reruns only what is missing.
#
# Landsat 7 out after 2021 (AMBIGUITIES #30), as in the reads behind
# source_mpc_maize_2008_2023_l7_2021.parquet: 2008-2021 read as the 2008-2018 Iowa reads did.
# %50 caps the MPC readers at 50 in all. Iowa ran at 25, a number never measured against
# MPC's throttling: grep the first logs for 429 and timeouts, and if they show up, lower the
# cap with `scontrol update ArrayTaskThrottle=25 JobId=<job>`.
#
# Once, on a login node, after pulling the commit that adds this file:
#   apptainer exec py.sif uv sync --frozen
#   # CropScape's CDL for 2022-2023 (mpc.MPC_CDL_LAST), fetched once here rather than by
#   # every task at once:
#   apptainer exec py.sif .venv/bin/python -c "from qdann import mpc; [mpc.cropscape(y, s) for y in (2022, 2023) for s in ('20', '29', '46')]"
#   # data/ is not in git: copy the laptop's list into the same place
#   scp data/stage2/reads.txt <sherlock>:<repo>/data/stage2/
#   n=$(wc -l < data/stage2/reads.txt)
#   mkdir -p logs && sbatch --array=0-$((n - 1))%50 sherlock_state.sbatch
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
read -r fips ranges < <(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" data/stage2/reads.txt)
out="$SCRATCH/qdann/mpc_l7_2021/${fips:0:2}/$fips"
for years in $ranges; do
    $python -m qdann.mpc --crop maize --years "$years" --county "$fips" --out "$out" \
        --landsat7-last 2021
done
