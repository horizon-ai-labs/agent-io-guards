#!/bin/bash
# usage: bash fin/run_teacher_eval.sh -> $OUTPUT_DIR/teacher_fin2.json
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/teacher.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
for v in fin2; do
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 PROMPT_V=$v python sentiment/teacher_sent.py $OUTPUT_DIR/teacher_$v.json fin/evals 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
done
