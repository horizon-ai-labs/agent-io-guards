#!/bin/bash
# usage: BASE=... EXTRA="..." [N=60000] bash safety/run_train_v3.sh  (v2 data (ALPHA=1, job dcaf7c1c) + Civil Comments; train; eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
R=$JOBS
export HF_HOME=${TMPDIR:-/tmp}/hf
python safety/build_safety_v3.py $R/dcaf7c1c-1644-4e9e-9834-54df343fc8f6/safety_v2 $WORKSPACE/safety/evals $OUTPUT_DIR/safety_v3 ${N:-60000} 2>&1 | tee $OUTPUT_DIR/logs/build.log
unset HF_HOME
python safety/train_safety.py --base $BASE --data $OUTPUT_DIR/safety_v3 --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python safety/eval_safety.py --models $OUTPUT_DIR/model --evals $WORKSPACE/safety/evals --extra $R/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0/test.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
