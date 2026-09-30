#!/bin/bash
# usage: DATA=<lid data dir> BASE=... EXTRA="..." bash lid/run_train.sh  (train, then FLORES eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
python lid/train_lid.py --base $BASE --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python lid/eval_lid.py --models $OUTPUT_DIR/model --evals $DATA/eval --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
