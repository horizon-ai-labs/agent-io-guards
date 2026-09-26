#!/bin/bash
# usage: bash safety/run_pool.sh -> $OUTPUT_DIR/pool.parquet (raw) and $OUTPUT_DIR/pool_labelled.parquet (Qwen3Guard-Gen-8B)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
export HF_HOME=${TMPDIR:-/tmp}/hf
python safety/build_prompt_pool.py $OUTPUT_DIR/pool.parquet 2>&1 | tee $OUTPUT_DIR/logs/pool.log
unset HF_HOME
python safety/label_teacher.py $OUTPUT_DIR/pool.parquet $OUTPUT_DIR/pool_labelled.parquet ${TEACHER:-Qwen/Qwen3Guard-Gen-8B} 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/label.log
