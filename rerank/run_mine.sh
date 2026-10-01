#!/bin/bash
# usage: QDIR=<dir with q_*.jsonl> bash rerank/run_mine.sh -> $OUTPUT_DIR/mined.parquet
set -o pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
PYTHONDONTWRITEBYTECODE=1 python rerank/mine.py $WORKSPACE/.gated/rr/passages.parquet $QDIR $OUTPUT_DIR/mined.parquet 24 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/mine.log
[ -s $OUTPUT_DIR/mined.parquet ] || exit 5
