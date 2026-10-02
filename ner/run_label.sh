#!/bin/bash
# usage: SHARD=i NSHARD=n bash ner/run_label.sh -> $OUTPUT_DIR/ner_<shard>.jsonl (Qwen NER labels, prompt v2, on rerank passages)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PROMPT_V=2 IN=$WORKSPACE/.gated/rr/passages.parquet OUT=$OUTPUT_DIR/ner_$SHARD.jsonl python ner/teacher_ner.py label 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/ner_$SHARD.jsonl ] || exit 5
