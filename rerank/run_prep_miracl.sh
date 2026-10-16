#!/bin/bash
# usage: bash rerank/run_prep_miracl.sh -> $OUTPUT_DIR/miracl/{miracl_q,miracl_passages}.parquet  (CPU job; corpus files downloaded to node-local /tmp)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
HF_HOME=${TMPDIR:-/tmp}/hf CLEAN=1 python rerank/prep_miracl.py $OUTPUT_DIR/miracl 2>&1 | grep -v Warn | tee $OUTPUT_DIR/logs/prep.log
