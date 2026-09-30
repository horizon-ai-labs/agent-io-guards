#!/bin/bash
# usage: V4=<lid_v3 dir from the v4 job> BASE=... EXTRA="..." bash lid/run_v5.sh  (merge labels -> lid_v5, train, eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python lid/merge_labels.py $V4 $OUTPUT_DIR/lid_v5 2>&1 | tee $OUTPUT_DIR/logs/merge.log
DATA=$OUTPUT_DIR/lid_v5 bash lid/run_train.sh
