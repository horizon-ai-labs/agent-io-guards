#!/bin/bash
# usage: bash sentiment/run_teacher_v2c.sh -> $OUTPUT_DIR/teacher_v2_mteb.json (prompt v2 on the mteb sets)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/teacher.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PROMPT_V=2 ONLY=mteb python sentiment/teacher_sent.py $OUTPUT_DIR/teacher_v2_mteb.json sentiment/evals 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/teacher_v2_mteb.json ] || exit 5
