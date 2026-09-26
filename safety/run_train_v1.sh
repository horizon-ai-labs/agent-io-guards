#!/bin/bash
# usage: POOL=<labelled pool parquet> BASE=... EXTRA="..." [FRAC=1.0] bash safety/run_train_v1.sh  (build v1, train, eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
V0=$JOBS/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0
python safety/build_safety_v1.py $V0 $POOL $WORKSPACE/safety/evals $OUTPUT_DIR/safety_v1 ${FRAC:-1.0} 2>&1 | tee $OUTPUT_DIR/logs/build.log
python safety/train_safety.py --base $BASE --data $OUTPUT_DIR/safety_v1 --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python safety/eval_safety.py --models $OUTPUT_DIR/model --evals $WORKSPACE/safety/evals --extra $V0/test.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
