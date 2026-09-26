#!/bin/bash
# usage: MODELS=a,b[,qwen3guard:Qwen/Qwen3Guard-Gen-0.6B] bash safety/run_eval.sh
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
DATA=$JOBS/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0
python safety/eval_safety.py --models $MODELS --evals $WORKSPACE/safety/evals --extra $DATA/test.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
