#!/bin/bash
# usage: IN=<mined/subset parquet> SHARD=i NSHARD=n [TEACHER=...] bash rerank/run_score.sh -> $OUTPUT_DIR/scored_<shard>.parquet
set -o pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
PYTHONDONTWRITEBYTECODE=1 PASSAGES=$WORKSPACE/.gated/rr/passages.parquet OUT=$OUTPUT_DIR/scored_$SHARD.parquet python rerank/teacher_score.py 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/score.log
[ -s $OUTPUT_DIR/scored_$SHARD.parquet ] || exit 5
