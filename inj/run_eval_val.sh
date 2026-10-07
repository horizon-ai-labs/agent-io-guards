#!/bin/bash
# usage: MODELS=dir bash inj/run_eval_val.sh  (per-item predictions on the v2 validation split, for threshold calibration)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python train/evaluate.py --normalize --evals $WORKSPACE/.gated/inj/v2val --models $MODELS --out $OUTPUT_DIR/eval_results.json 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/eval.log
