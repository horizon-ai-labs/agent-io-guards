#!/bin/bash
# usage: bash scam/run_teacher_eval.sh -> $OUTPUT_DIR/teacher.json (Qwen3.8-27B on scam/evals)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/teacher.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 python scam/teacher_scam.py $OUTPUT_DIR/teacher.json scam/evals 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
