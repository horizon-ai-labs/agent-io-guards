#!/bin/bash
# usage: IN=<parquet with text,text_pair> SHARD=i NSHARD=n [TEACHER=...] bash safety/run_label.sh -> $OUTPUT_DIR/labelled.parquet
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
if [ "$IN" = pool2 ]; then
  python safety/build_pool2.py $JOBS/65d4f827-c645-4a38-bf33-8d5ea76adb08/v2 $OUTPUT_DIR/pool2.parquet 2>&1 | tee $OUTPUT_DIR/logs/pool2.log
  IN=$OUTPUT_DIR/pool2.parquet
fi
python safety/label_teacher.py $IN $OUTPUT_DIR/labelled.parquet ${TEACHER:-Qwen/Qwen3Guard-Gen-8B} 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/label.log
