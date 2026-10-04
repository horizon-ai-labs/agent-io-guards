#!/bin/bash
# usage: MODELS=a,b bash fin/run_eval_baselines.sh -> $OUTPUT_DIR/eval_results.json (fin/evals, via sentiment/eval_sent.py)
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
python3 sentiment/eval_sent.py --models $MODELS --evals fin/evals --out $OUTPUT_DIR/eval_results.json 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/eval.log
