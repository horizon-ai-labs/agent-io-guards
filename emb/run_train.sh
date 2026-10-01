#!/bin/bash
# usage: TEACHER=<dir[,dir]> BASE=... EXTRA="..." bash emb/run_train.sh  (distil bge-m3, then cosine-reranking eval: symmetric + asymmetric)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python emb/train_emb.py --base ${BASE:-jhu-clsp/mmBERT-small} --teacher $TEACHER --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python emb/eval_emb.py --models ours:$OUTPUT_DIR/model,asym:$OUTPUT_DIR/model --evals rerank/evals --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
