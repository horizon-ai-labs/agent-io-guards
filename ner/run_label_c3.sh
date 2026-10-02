#!/bin/bash
# usage: SHARD=i NSHARD=n bash ner/run_label_tail.sh -> $OUTPUT_DIR/ner_c3_<shard>.jsonl (chunk 3 of main shards 1, 2, 4 (insurance against the 3h limit))
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn PROMPT_V=2 IN=$WORKSPACE/.gated/ner/passages_c3.parquet OUT=$OUTPUT_DIR/ner_c3_$SHARD.jsonl python ner/teacher_ner.py label 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/ner_c3_$SHARD.jsonl ] || exit 5
