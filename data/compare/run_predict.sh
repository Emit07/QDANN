#!/bin/bash
# Step 5 reads: 10 central-Iowa counties x {2018, 2016}, 3 counties per run (memory).
# Each group writes under its own --out, so the per-run 5' frames and json don't collide.
set -euo pipefail
cd "$(dirname "$0")/../.."
groups=("19169 19015 19153" "19127 19079 19083" "19099 19049 19187" "19075")
for year in 2018 2016; do
  for i in "${!groups[@]}"; do
    out=data/compare/predict/g$i
    if ls $out/*/features_maize_$year.npz >/dev/null 2>&1 && \
       [ "$(ls $out/*/features_maize_$year.npz | wc -l)" -eq "$(wc -w <<< "${groups[$i]}")" ]; then
      echo "skip $year g$i"; continue
    fi
    echo "== $year g$i: ${groups[$i]} $(date +%T)"
    uv run python -m qdann.predict --crop maize --year $year --county ${groups[$i]} --out $out
  done
done
echo "all done $(date +%T)"
