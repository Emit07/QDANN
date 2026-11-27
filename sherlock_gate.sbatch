#!/bin/bash
# Stage 1 gate for 2020, 2022 and 2023 on the 25 gate counties (notes/yield_layer_plan.md,
# Step 8 results): read each county-year's crop-pixel series, fit its pixel features, keep
# only the features (~40 MB; the ~300 MB series is deleted). Training and scoring stay on
# the laptop: copy the features back into data/gate/features and run gate.py there.
# One array task per county-year. Resumes: read.py skips series on disk, and a task whose
# features exist exits at once.
#
# Same reader as the 2008-2023 table (#27): Landsat 7 out after 2021 (#30), CDL 2022+ from
# the CropScape files already copied for sherlock_extend.sbatch (data/cdl/CDL_202{2,3}_19.tif).
#
# Once, on a login node, after pulling the commit that adds this file:
#   mkdir -p $SCRATCH/qdann/gate/{series,features} logs
#   ln -sfn $SCRATCH/qdann/gate/series data/gate/series      # off $HOME's quota
#   ln -sfn $SCRATCH/qdann/gate/features data/gate/features
#   # gridMET once, so 75 tasks don't race to download the same files
#   apptainer exec py.sif .venv/bin/python -c "from qdann import weather; [weather.gridmet_file(v, y) for v in weather.VARIABLES for y in (2020, 2022, 2023)]"
#   sbatch sherlock_gate.sbatch
# Then, from the laptop:
#   rsync -av sherlock:'$SCRATCH/qdann/gate/features/*_202[023].parquet' data/gate/features/
#
#SBATCH --job-name=qdann-gate
#SBATCH --partition=normal,owners
#SBATCH --array=0-74%25
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=06:00:00
#SBATCH --output=logs/gate_%a_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="apptainer exec py.sif .venv/bin/python"
counties=(19169 19015 19153 19127 19079 19083 19099 19049 19187 19075
          19011 19037 19059 19071 19081 19087 19103 19111 19115 19119 19137 19139 19143 19145 19165)
years=(2020 2022 2023)
fips=${counties[$((SLURM_ARRAY_TASK_ID % 25))]}
year=${years[$((SLURM_ARRAY_TASK_ID / 25))]}
# fail fast on missing shared inputs: a missing CDL or gridMET file would be downloaded by
# every task at once, racing on the same .part file
for f in data/source_mpc_maize_2008_2023_l7_2021.parquet data/tune_maize_2008_2023_l7_2021.json \
         data/cdl/CDL_2022_19.tif data/cdl/CDL_2023_19.tif \
         data/gridmet/{pr,srad,tmmn,tmmx,vpd}_{2020,2022,2023}.nc; do
    [ -s "$f" ] || { echo "missing $f: run the setup in this file's header"; exit 1; }
done
feat=data/gate/features/${fips}_${year}.parquet
[ -e "$feat" ] && { echo "have $feat"; exit 0; }
$python data/gate/read.py --year "$year" --county "$fips" --landsat7-last 2021
# GATE_TABLE: gate.py reads that table and its tune json at import (both on Sherlock from
# sherlock_extend_fit.sbatch); all_features writes the cache
GATE_TABLE=2008_2023_l7_2021 $python -c "
import sys; sys.path.insert(0, 'data/gate'); import gate
assert gate.all_features('$fips', $year) is not None, 'no series for $fips $year'"
rm data/gate/series/${fips}_${year}.npz
echo "done $fips $year"
