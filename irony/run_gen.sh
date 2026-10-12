#!/bin/bash
# usage: SHARD=i N=3000 bash irony/run_gen.sh -> $OUTPUT_DIR/gen/irony_<shard>.jsonl
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/gen.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 OUT=$OUTPUT_DIR/gen python irony/gen_irony.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
