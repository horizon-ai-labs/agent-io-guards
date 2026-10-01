#!/bin/bash
# usage: DATA=<scored parquet(s), comma-separated> BASE=... EXTRA="..." bash rerank/run_train.sh  (train, then reranking benchmarks)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python rerank/train_rr.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --passages $WORKSPACE/.gated/rr/passages.parquet --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python rerank/eval_rerank.py --models $OUTPUT_DIR/model --evals rerank/evals --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
