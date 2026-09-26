#!/bin/bash
# usage: BASE=... [DATA=...] EXTRA="..." bash safety/run_train.sh   (train, then external + in-domain eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
DATA=${DATA:-$JOBS/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0}
python safety/train_safety.py --base $BASE --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python safety/eval_safety.py --models $OUTPUT_DIR/model --evals $WORKSPACE/safety/evals --extra $DATA/test.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
