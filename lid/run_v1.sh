#!/bin/bash
# usage: V0=<lid_v0 dir> MODEL0=<v0 model dir> BASE=... EXTRA="..." bash lid/run_v1.sh  (clean -> lid_v1, train, eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python lid/clean_lid.py $MODEL0 $V0 $OUTPUT_DIR/lid_v1 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/clean.log
DATA=$OUTPUT_DIR/lid_v1 bash lid/run_train.sh
