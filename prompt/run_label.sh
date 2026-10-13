#!/bin/bash
# usage: LABEL_IN=... LABEL_NAME=... SHARD=i NSHARD=n [JUDGE=1] bash prompt/run_label.sh -> $OUTPUT_DIR/${LABEL_NAME}_$SHARD.parquet
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
[ -n "$JUDGE" ] && export MODEL=google/gemma-4-31b-it QUANT=fp8 HF_HOME=${TMPDIR:-/tmp}/hf
LABEL_OUT=$OUTPUT_DIR/${LABEL_NAME:-lab}_${SHARD:-0}.parquet VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 \
  python prompt/teacher_prompt.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
