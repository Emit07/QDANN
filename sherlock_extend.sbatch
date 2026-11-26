#!/bin/bash
# Extend the source table to 2019-2023 (notes/yield_layer_plan.md, Step 7 "Decided"):
# MPC county means for every Iowa county, one array task per county, county means only (no
# target pixels). Same container and venv as sherlock.sbatch. Resumes by skipping years on
# disk, so resubmitting the same array reruns only what is missing.
#
# Reads go to their own directory, so the 2008-2018 reads under mpc/19 are never refitted
# or mixed in: the 2008-2018 rows come unchanged from data/source_mpc_maize.parquet.
# A scene whose assets MPC serves truncated (11 in 2022 over Iowa) is dropped with a
# "skipping unreadable scene" warning: grep logs/ext_*.out for it afterwards.
#
# Once, on a login node, after pulling the commit that adds this file:
#   apptainer exec py.sif uv sync --frozen
#   apptainer exec py.sif git rev-parse HEAD          # tune needs git in the container
#   # from the laptop: the 2008-2018 table, the 2019-2023 labels and weather, and CropScape's
#   # CDL for 2022-2023 (mpc.MPC_CDL_LAST; else 99 tasks would each try to fetch it)
#   scp data/source_mpc_maize.parquet sherlock:<repo>/data/
#   scp data/extend/{nass,gridmet}_maize_2019_2023.parquet sherlock:<repo>/data/extend/
#   scp data/cdl/CDL_202{2,3}_19.tif sherlock:<repo>/data/cdl/
#
# Landsat 7 (AMBIGUITIES #30) must be chosen before the read: on shared dates its pixels are
# mosaicked with Landsat 8/9's, so it can't be dropped afterwards. Both jobs need it:
#   export L7_LAST=2021   # leave Landsat 7 out from 2022 (recommended), or
#   export L7_LAST=all    # keep it every year (the paper's reader)
#   mkdir -p logs && jid=$(sbatch --parsable sherlock_extend.sbatch)
#   sbatch --dependency=afterok:$jid sherlock_extend_fit.sbatch
#
# afterok: if any task fails, the fit job waits forever. Resubmit the array, then cancel
# the fit job and resubmit it against the new array's id.
#
#SBATCH --job-name=qdann-ext
#SBATCH --partition=normal,owners
#SBATCH --array=0-98%25
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=12:00:00
#SBATCH --output=logs/ext_%a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
L7_LAST=${L7_LAST:?export L7_LAST=2021 or L7_LAST=all first (AMBIGUITIES #30)}
# the choice names the directory, so skip-if-exists never reuses reads made under the other
out="$SCRATCH/qdann/mpc_ext_l7_$L7_LAST/19"
python="apptainer exec py.sif .venv/bin/python"
l7=$([ "$L7_LAST" = all ] || echo "--landsat7-last $L7_LAST")
fips=$($python -c "from qdann import mpc; print(sorted(mpc.state_counties('19'))[$SLURM_ARRAY_TASK_ID])")
exec $python -m qdann.mpc --crop maize --years 2019-2023 --county "$fips" --out "$out/$fips" $l7
