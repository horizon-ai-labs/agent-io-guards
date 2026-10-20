#!/bin/bash
# usage: bash rerank/run_prep_webfaq.sh -> $OUTPUT_DIR/webfaq/{webfaq_passages.parquet, q/q_webfaq.jsonl}  (CPU job)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
HF_HOME=${TMPDIR:-/tmp}/hf python rerank/prep_webfaq.py $OUTPUT_DIR/webfaq 2>&1 | grep -v Warn | tee $OUTPUT_DIR/logs/prep.log
