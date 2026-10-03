#!/bin/bash
# usage: PREP=<prep dir> MODELS="dir1 dir2" bash punct/run_eval_models.sh
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
python3 punct/eval_punct.py $OUTPUT_DIR/eval_results.json $PREP/evals $MODELS 2>&1 | tee $OUTPUT_DIR/logs/eval.log
