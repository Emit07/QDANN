#!/bin/bash
# Overnight 2026-11-27: widen the 2019/2021 gate for the 2008-2023 heads (plan Step 8).
# Serial on purpose (7 GB RAM): reads, then leave-one-county-out heads, then scoring.
# Resumable: read.py skips county-years on disk, gate.py train skips heads on disk.
cd "$(dirname "$0")/../.."
wide=("19011 19037 19059" "19071 19081 19087" "19103 19111 19115" "19119 19137 19139" "19143 19145 19165")
run() { echo "== read $1: $2 $(date +%T)"; uv run python data/gate/read.py --year $1 --county $2 || echo "FAILED $1 $2"; }
for y in 2019 2021; do for g in "${wide[@]}"; do run $y "$g"; done; done
export GATE_TABLE=2008_2023_l7_2021
echo "== train $(date +%T)"
uv run python data/gate/gate.py train ${wide[*]} || echo "FAILED train"
for f in 15 20; do
  echo "== score $f $(date +%T)"
  uv run python data/gate/gate.py score $f > data/gate/score_f${f}_ext_wide.log 2>&1 || echo "FAILED score $f"
done
echo "overnight done $(date +%T)"
