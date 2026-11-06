#!/bin/bash
# usage: IN=<pool parquet> MODEL_DIR=<gemma job-id>/model bash ncii/run_judge.sh -> $OUTPUT_DIR/judged.parquet
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/judge.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
export MODEL=$(dirname $OUTPUT_DIR)/$MODEL_DIR HF_HUB_OFFLINE=1 QUANT=${QUANT:-fp8}
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn OUT=$OUTPUT_DIR/judged.parquet python ncii/judge_ncii.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/judged.parquet ] || exit 5
