#!/bin/bash
# usage: SRC_JOB=<training job id> bash inj/run_release26.sh -> $OUTPUT_DIR/model (release dir: weights, tokenizer, ONNX fp32 + int8)
# then the as-shipped eval of the release dir on all injection evals incl. .gated/eval_tool (release normalizer).
set -eo pipefail
cd $WORKSPACE; R=$(dirname $OUTPUT_DIR); V2=$R/65d4f827-c645-4a38-bf33-8d5ea76adb08/v2
SRC=$R/$SRC_JOB/model BASE=jhu-clsp/mmBERT-small EV=$V2/eval bash release/run_release_build.sh
source scripts/env.sh
python train/evaluate.py --normalize --evals $V2/eval,$WORKSPACE/.gated/eval,$WORKSPACE/.gated/eval_tool --models $OUTPUT_DIR/model --out $OUTPUT_DIR/eval_results.json 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/eval.log
