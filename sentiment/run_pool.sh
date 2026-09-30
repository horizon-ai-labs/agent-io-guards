#!/bin/bash
# usage: bash sentiment/run_pool.sh  (CPU job) -> $OUTPUT_DIR/pool.parquet
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/pool.log) 2>&1
source scripts/env.sh
NPROC=14 python sentiment/build_pool.py $OUTPUT_DIR/pool.parquet 2500
