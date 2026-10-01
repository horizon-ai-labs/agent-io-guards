#!/bin/bash
# usage: bash rerank/run_passages.sh  (CPU job) -> $OUTPUT_DIR/passages.parquet
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/passages.log) 2>&1
source scripts/env.sh
NPROC=14 python rerank/build_passages.py $OUTPUT_DIR/passages.parquet 8000
