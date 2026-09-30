#!/bin/bash
# usage: LABEL_IN=<parquet> SHARD=i NSHARD=n bash sentiment/run_label.sh -> $OUTPUT_DIR/labelled_<shard>.parquet (teacher soft labels)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/label.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
VLLM_WORKER_MULTIPROC_METHOD=spawn LABEL_OUT=$OUTPUT_DIR/labelled_$SHARD.parquet python sentiment/teacher_sent.py - - 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/labelled_$SHARD.parquet ] || exit 5
