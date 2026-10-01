#!/bin/bash
# usage: bash emb/run_texts.sh  (CPU job) -> $OUTPUT_DIR/texts.parquet
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/texts.log) 2>&1
source scripts/env.sh
NPROC=15 python emb/build_texts.py $OUTPUT_DIR/texts.parquet 36000
