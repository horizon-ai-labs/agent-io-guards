#!/bin/bash
# usage: [LABEL_IN=... LABEL_NAME=...] SHARD=i NSHARD=n bash hate/run_label.sh -> $OUTPUT_DIR/lab_$SHARD.parquet (hate + 7 target teacher labels for .gated/hate/civil_pool.parquet)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
LABEL_IN=${LABEL_IN:-$WORKSPACE/.gated/hate/civil_pool.parquet} LABEL_OUT=$OUTPUT_DIR/${LABEL_NAME:-lab}_$SHARD.parquet \
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 python hate/teacher_hate.py label 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
