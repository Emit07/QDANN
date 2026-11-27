#!/bin/bash
# Overnight 2026-11-25 read queue, one MPC read at a time (memory). Resumable: read.py skips
# county-years on disk. Order: floor study (2018 core), gate breadth (2018), unseen years.
cd "$(dirname "$0")/../.."
core=("19169 19015 19153" "19127 19079 19083" "19099 19049 19187" "19075")
wide=("19011 19037 19059" "19071 19081 19087" "19103 19111 19115" "19119 19137 19139" "19143 19145 19165")
run() { echo "== $1: $2 $(date +%T)"; uv run python data/gate/read.py --year $1 --county $2 || echo "FAILED $1 $2"; }
for g in "${core[@]}"; do run 2018 "$g"; done
for g in "${wide[@]}"; do run 2018 "$g"; done
for y in 2019 2021; do for g in "${core[@]}"; do run $y "$g"; done; done
echo "queue done $(date +%T)"
