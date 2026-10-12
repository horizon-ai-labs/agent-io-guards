#!/bin/bash
# usage: DOCS=<test_docs.parquet> bash edu/run_judge.sh -> $OUTPUT_DIR/judge.json (Gemma-4-31B vs Llama-3 annotations, English) and
#        $OUTPUT_DIR/test_judged.parquet (independent judge scores for the multilingual test documents)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/judge.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$(dirname $OUTPUT_DIR)/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
export MODEL=google/gemma-4-31b-it QUANT=fp8 HF_HOME=${TMPDIR:-/tmp}/hf VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1
python edu/teacher_edu.py eval $OUTPUT_DIR/judge.json edu/evals 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
LABEL_IN=$DOCS LABEL_OUT=$OUTPUT_DIR/test_judged.parquet python edu/teacher_edu.py label 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
