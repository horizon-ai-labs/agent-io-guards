#!/bin/bash
# usage: DATA=<dir> BASE=... EXTRA="..." bash hate/run_train.sh  (tox/train_tox.py with the hate labels, then hate/eval_hate.py)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
L=hate,target_race,target_religion,target_origin,target_gender,target_sexuality,target_age,target_disability
python tox/train_tox.py --labels $L --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python hate/eval_hate.py --models $OUTPUT_DIR/model --evals hate/evals --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
