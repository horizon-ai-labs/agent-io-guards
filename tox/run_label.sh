#!/bin/bash
# usage: [LABEL_IN=... LABEL_NAME=...] SHARD=i NSHARD=n bash tox/run_label.sh -> $OUTPUT_DIR/native_lab_$SHARD.parquet (7 Qwen3.8-27B teacher labels for .gated/tox/native.parquet)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
LABEL_IN=${LABEL_IN:-$WORKSPACE/.gated/tox/native.parquet} LABEL_OUT=$OUTPUT_DIR/${LABEL_NAME:-native_lab}_$SHARD.parquet \
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 python tox/teacher_tox.py label 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
