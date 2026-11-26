#!/bin/bash
# usage: BASE=... DATA=<scored dir with soft col> EXTRA="--distill 0.5 ..." bash train/run_train_distill.sh
# trains, then evaluates with the release normalizer on the injection eval sets (same as train/run_eval_norm.sh)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
python train/train.py --base $BASE --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
V2=$(dirname $OUTPUT_DIR)/65d4f827-c645-4a38-bf33-8d5ea76adb08/v2
python train/evaluate.py --normalize --evals $V2/eval,$WORKSPACE/.gated/eval${EXTRA_EVALS:+,$EXTRA_EVALS} --models $OUTPUT_DIR/model --out $OUTPUT_DIR/eval_results.json 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/eval.log
