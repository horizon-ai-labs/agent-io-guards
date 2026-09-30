#!/bin/bash
# usage: bash sentiment/run_teacher_v2.sh -> $OUTPUT_DIR/teacher_v1.json, teacher_v2.json (tweets + amazon, with confusion counts)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/teacher.log) 2>&1
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
for v in 1 2; do
PROMPT_V=$v ONLY=tweets,amazon python sentiment/teacher_sent.py $OUTPUT_DIR/teacher_v$v.json sentiment/evals 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
done
