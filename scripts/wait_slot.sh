#!/bin/bash
# usage: bash scripts/wait_slot.sh [N]  -- wait (up to ~9 min) until fewer than N (default 8) jobs are active; exit 1 on timeout
n=${1:-8}
for i in $(seq 1 18); do
  a=$(compute status | python3 -c "import json,sys;print(json.load(sys.stdin)['active_jobs'])")
  [ "$a" -lt "$n" ] && exit 0; sleep 30
done; exit 1
