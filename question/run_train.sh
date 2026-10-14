#!/bin/bash
# usage: DATA=<dir> BASE=... EXTRA="..." bash question/run_train.sh  (train/train.py 2-class softmax, LABEL_0 = statement/keywords, LABEL_1 = question;
# the label convention of shahrukhx01/question-vs-statement-classifier for drop-in use), then question/eval_question.py
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python train/train.py --labels LABEL_0,LABEL_1 --max_len 128 --tokens_per_batch 32768 --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python question/eval_question.py --models $OUTPUT_DIR/model --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
