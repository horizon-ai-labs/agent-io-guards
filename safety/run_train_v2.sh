#!/bin/bash
# usage: ALPHA=1.0 BASE=... EXTRA="..." bash safety/run_train_v2.sh  (build v2, train, eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
R=$JOBS
V0=$R/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0
V0L=$R/dd140679-c3eb-49eb-8fca-974c9a32cc7d/labelled.parquet,$R/c34b5bda-6957-486d-a558-5c6cd217d0b9/labelled.parquet,$R/046d7149-ad8b-4213-909b-8e8878ff9452/labelled.parquet,$R/931519bf-e7ec-43bc-8f02-a0b51e5d16e5/labelled.parquet
P1=$R/691f6b5c-a835-4c5d-8cd3-d73598352cd4/pool_labelled.parquet
P2=$R/d4120ea3-5411-49eb-b77b-f2104691421d/labelled.parquet
ALPHA=${ALPHA:-1.0} python safety/build_safety_v2.py $V0 "$V0L" $P1 $P2 $WORKSPACE/safety/evals $OUTPUT_DIR/safety_v2 2>&1 | tee $OUTPUT_DIR/logs/build.log
python safety/train_safety.py --base $BASE --data $OUTPUT_DIR/safety_v2 --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python safety/eval_safety.py --models $OUTPUT_DIR/model --evals $WORKSPACE/safety/evals --extra $V0/test.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
