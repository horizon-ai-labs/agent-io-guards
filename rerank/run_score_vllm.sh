#!/bin/bash
# usage: IN=<parquet> SHARD=i NSHARD=n [LIMIT=n] bash rerank/run_score_vllm.sh -> $OUTPUT_DIR/scored_<shard>.parquet (written after every chunk)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/score.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PASSAGES=$WORKSPACE/.gated/rr/passages.parquet OUT=$OUTPUT_DIR/scored_$SHARD.parquet python rerank/teacher_score_vllm.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/scored_$SHARD.parquet ] || exit 5
