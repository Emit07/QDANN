#!/bin/sh
# Step 2 of the end-to-end plan: 2018 target pixels for three held-out counties
for fips in 19013 19083 19113; do
  uv run python -m qdann.mpc --crop maize --county $fips --years 2018 --target --out data/mpc/e2e/$fips || echo "FAILED $fips"
done
echo "ALL DONE"
