#!/bin/bash
# usage: DATA=<dir> BASE=... EXTRA="..." bash gib/run_train.sh  (lid/train_lid.py on the 4 gibberish classes, then eval incl. baseline)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python lid/train_lid.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
FL=$JOBS/58cb13ac-253c-43c8-927b-60505e273019/lid_v5/eval/flores_devtest.parquet
python gib/eval_gib.py --models $OUTPUT_DIR/model${BASELINE:+,$BASELINE} --test $DATA/test.parquet --flores $FL --hand gib/handset.json --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
