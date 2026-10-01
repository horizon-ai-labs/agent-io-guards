#!/bin/bash
# usage: DATA=<dir> BASE=jhu-clsp/mmBERT-small EXTRA="..." bash emotion/run_train.sh  (multi-label training, then go_emotions + BRIGHTER eval)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python emotion/train_emo.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python emotion/eval_emo.py --models $OUTPUT_DIR/model --evals emotion/evals --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
