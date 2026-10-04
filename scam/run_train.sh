#!/bin/bash
# usage: DATA=<dir> BASE=jhu-clsp/mmBERT-small EXTRA="..." bash scam/run_train.sh  (teacher soft labels -> scam/evals)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
LABELS=legitimate,spam,fraud PCOLS=p_legit,p_spam,p_fraud python sentiment/train_sent.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model --max_len 384 $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python scam/eval_scam.py $OUTPUT_DIR/eval.json scam/evals $OUTPUT_DIR/model 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
