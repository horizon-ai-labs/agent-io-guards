#!/bin/bash
# usage: JOBS="<job ids with model/>" bash inj/run_tool_eval.sh -> $OUTPUT_DIR/eval_results.json (+ preds/) on .gated/eval_tool
# (inj/build_tool_evals.py), with the release normalizer.
set -eo pipefail
cd $WORKSPACE; source scripts/env.sh; mkdir -p $OUTPUT_DIR/logs
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); M=$(for j in $JOBS; do printf "%s," "$R/$j/model"; done); M=${M%,}
python train/evaluate.py --normalize --evals $WORKSPACE/.gated/eval_tool --models $M --out $OUTPUT_DIR/eval_results.json 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/eval.log
