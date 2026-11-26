#!/bin/bash
# After sherlock_extend.sbatch: fit Eq. 2 to the 2019-2023 reads, join them to the labels
# and weather into the 2008-2023 table named below, then rerun Stage 0 (`tune`) on
# that table: grid, held-out folds, leave-one-year-out, baselines, and one model on every
# county, saved as data/models/qdann_maize_2008_2023_l7_<L7_LAST>.pt with
# tune_maize_2008_2023_l7_<L7_LAST>.json (table: source_mpc_maize_2008_2023_l7_<L7_LAST>).
# Nothing here carries the shipped names; shipping it is Ale's call (CLAUDE.md).
#
#   sbatch --dependency=afterok:$jid sherlock_extend_fit.sbatch
#
# Resubmitting after a timeout resumes tune from data/tune_maize.jsonl. If any county was
# re-read after the table was built, delete that table first:
# an existing table is reused as is.
#
# `normal` only: a preempted owners job requeued after the save would stop on "exists".
#SBATCH --job-name=qdann-ext-fit
#SBATCH --partition=normal
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH --time=8:00:00
#SBATCH --output=logs/ext_fit_%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
python="apptainer exec py.sif .venv/bin/python"
L7_LAST=${L7_LAST:-2021}  # as in sherlock_extend.sbatch
reads="$SCRATCH/qdann/mpc_ext_l7_$L7_LAST/19"
table=data/source_mpc_maize_2008_2023_l7_$L7_LAST.parquet
# tune records the git commit; fail now, not after hours, if the container can't see it
$python -c "from qdann import tune; print(tune.provenance())"
if [ ! -e "$table" ]; then
    $python -m qdann.mpc --crop maize --fit "$reads"
    $python data/extend/join.py "$reads/gcvi_mpc_maize.parquet" "$table"
fi
echo "CPUs: $($python -c 'import os; print(os.process_cpu_count())')"
exec $python -m qdann.tune --crop maize --source "$table" --name "2008_2023_l7_$L7_LAST"
