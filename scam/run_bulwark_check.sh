#!/bin/bash
# usage: JOBS="<release build job ids>" bash scam/run_bulwark_check.sh -> $OUTPUT_DIR/{eval_results,teacher}.json + dumps/
# Our models + the Qwen teacher on the third-party Bulwark Bench TR (scam/build_bulwark_eval.py).
cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs $OUTPUT_DIR/dumps
exec > >(tee $OUTPUT_DIR/logs/check.log) 2>&1
python3 -c 'import torch,sys; torch.zeros(1).cuda()' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); MODELS=""; for j in $JOBS; do MODELS="$MODELS $R/$j/model"; done
DUMP=$OUTPUT_DIR/dumps python3 scam/eval_scam.py $OUTPUT_DIR/eval_results.json scam/evals_bulwark $MODELS 2>&1 | grep --line-buffered -v Warning
VLLM_TAR=$R/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1 python scam/teacher_scam.py $OUTPUT_DIR/teacher.json scam/evals_bulwark 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
