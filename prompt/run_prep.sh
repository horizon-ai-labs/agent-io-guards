#!/bin/bash
# usage: bash prompt/run_prep.sh -> $OUTPUT_DIR/prompts/{train,test}_prompts.parquet  (CPU job)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
HF_HOME=${TMPDIR:-/tmp}/hf python prompt/prep_prompts.py $OUTPUT_DIR/prompts 2>&1 | tee $OUTPUT_DIR/logs/prep.log
