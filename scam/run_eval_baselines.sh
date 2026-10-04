#!/bin/bash
# usage: MODELS="a b@1+3" bash scam/run_eval_baselines.sh -> $OUTPUT_DIR/eval_results.json
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
python3 scam/eval_scam.py $OUTPUT_DIR/eval_results.json scam/evals $MODELS 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/eval.log
