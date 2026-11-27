#!/bin/sh
# One-off: the Tier B MPC reads, one county-year at a time (7 GB RAM); resumes by skipping done years.
cd "$(dirname "$0")/.."
while read fips year; do
  uv run python -m qdann.mpc --crop maize --county "$fips" --years "$year" || echo "FAILED $fips $year"
done < data/tier_b_subset.txt
echo ALL DONE
