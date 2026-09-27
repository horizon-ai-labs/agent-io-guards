#!/bin/bash
# usage: POOL=<labelled edgy parquet> BASE=... EXTRA="..." bash safety/run_train_v5.sh  (v4 data + pool; train; eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
R=$JOBS
python safety/build_add_pool.py $R/814efa37-9936-4afc-8a17-32820be38414/safety_v4 $POOL $WORKSPACE/safety/evals $OUTPUT_DIR/safety_v5 2>&1 | tee $OUTPUT_DIR/logs/build.log
python safety/train_safety.py --base $BASE --data $OUTPUT_DIR/safety_v5 --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python safety/eval_safety.py --models $OUTPUT_DIR/model --evals $WORKSPACE/safety/evals --extra $R/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0/test.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
