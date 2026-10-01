#!/bin/bash
# usage: MODELS=a,b [OUT=name] bash emb/run_eval.sh -> $OUTPUT_DIR/${OUT:-eval}.json
set -o pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
PYTHONDONTWRITEBYTECODE=1 python emb/eval_emb.py --models $MODELS --evals rerank/evals --out $OUTPUT_DIR/${OUT:-eval}.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee -a $OUTPUT_DIR/logs/eval.log
