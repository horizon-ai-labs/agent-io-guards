#!/bin/bash
# usage: V2=<lid_v1 dir from the v2 cleaning job> BASE=... EXTRA="..." bash lid/run_v3.sh  (augment short spans, train, eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python lid/augment_short.py $V2 $OUTPUT_DIR/lid_v3 2>&1 | tee $OUTPUT_DIR/logs/augment.log
DATA=$OUTPUT_DIR/lid_v3 bash lid/run_train.sh
