#!/bin/bash
# usage: SHARD=i N=4000 bash punct/run_gen_spoken.sh -> $OUTPUT_DIR/gen/spoken_<shard>.jsonl
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/gen.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 OUT=$OUTPUT_DIR/gen python punct/gen_spoken.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
