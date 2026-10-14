#!/bin/bash
# usage: LABEL_IN=... LABEL_NAME=... SHARD=i NSHARD=n [JUDGE=1] bash webtax/run_label.sh -> $OUTPUT_DIR/${LABEL_NAME}_$SHARD.parquet
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
if [ -n "$JUDGE" ]; then export MODEL=google/gemma-4-31b-it QUANT=fp8 HF_HOME=${TMPDIR:-/tmp}/hf
elif [ -n "$MODEL_DIR" ]; then export MODEL=$(dirname $OUTPUT_DIR)/$MODEL_DIR HF_HUB_OFFLINE=1; fi  # MODEL_DIR=<job-id>/model from scripts/fetch_model.sh (no Hub download)
LABEL_OUT=$OUTPUT_DIR/${LABEL_NAME:-lab}_${SHARD:-0}.parquet VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 \
  python webtax/teacher_webtax.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
