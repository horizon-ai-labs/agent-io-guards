#!/bin/bash
# usage: SHARD=i N=100 bash sentiment/run_gen2.sh -> $OUTPUT_DIR/gen/gen_<shard>.jsonl  (run_gen.sh + fail fast on an unusable GPU)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/gen.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
VLLM_WORKER_MULTIPROC_METHOD=spawn OUT=$OUTPUT_DIR/gen python sentiment/gen_sent.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
ls $OUTPUT_DIR/gen/*.jsonl >/dev/null || exit 5
