#!/bin/bash
# CPU job: training windows + evaluation sets -> $OUTPUT_DIR/{data,evals}
set -e; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 punct/build_evals.py $OUTPUT_DIR/evals 2>&1 | tee $OUTPUT_DIR/logs/evals.log
python3 punct/build_data.py $WORKSPACE/.gated/emb_texts.parquet $OUTPUT_DIR/data 2>&1 | tee $OUTPUT_DIR/logs/data.log
