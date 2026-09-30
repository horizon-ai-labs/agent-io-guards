#!/bin/bash
# usage: bash scripts/submit_label_v2.sh NAME LABEL_IN NSHARD SHARDS...  (teacher prompt v2 labelling shards; prints job ids)
name=$1; in=$2; ns=$3; shift 3
for s in "$@"; do
  compute request --json "{\"cpus\":8,\"memory_gb\":120,\"disk_gb\":60,\"gpus\":1,\"max_hours\":2,\"bid\":55,\"command\":\"PYTHONDONTWRITEBYTECODE=1 PROMPT_V=2 LABEL_IN=$in SHARD=$s NSHARD=$ns bash sentiment/run_label.sh\"}" | python3 -c "import json,sys;d=json.load(sys.stdin);print('$name', $s, d.get('id', d))"
done
