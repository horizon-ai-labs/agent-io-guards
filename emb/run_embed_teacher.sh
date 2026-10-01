#!/bin/bash
# usage: SRCS="a.parquet b.parquet:query" bash emb/run_embed_teacher.sh -> $OUTPUT_DIR/teacher/{texts.parquet,emb.npy}
set -o pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
PYTHONDONTWRITEBYTECODE=1 python emb/embed_teacher.py $OUTPUT_DIR/teacher $SRCS 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/embed.log
