#!/bin/bash
# usage: bash edu/run_prep.sh -> $OUTPUT_DIR/fw/{train_docs,test_docs}.parquet  (CPU job)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python edu/prep_fw.py $OUTPUT_DIR/fw 2>&1 | tee $OUTPUT_DIR/logs/prep.log
