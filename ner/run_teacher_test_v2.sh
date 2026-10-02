#!/bin/bash
# usage: bash ner/run_teacher_eval.sh -> $OUTPUT_DIR/teacher_v2_test.json (prompt v2 on test, for the card only; v2 was chosen on dev)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/teacher.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PROMPT_V=2 python ner/teacher_ner.py eval $OUTPUT_DIR/teacher_v2_test.json ner/evals 300 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/teacher_v2_test.json ] || exit 5
