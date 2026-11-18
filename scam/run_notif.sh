#!/bin/bash
# usage: SHARD=0 N=32 bash scam/run_notif.sh -> $OUTPUT_DIR/gen/notif_<shard>.jsonl, $OUTPUT_DIR/labelled_<shard>.parquet (teacher p_*)
# v1.1 data: scam/gen_notif.py (real-style genuine notifications + same-sector fraud), then the Qwen teacher labels them.
cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/notif.log) 2>&1
python3 -c "import torch; torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); SHARD=${SHARD:-0}
VLLM_TAR=$R/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
export VLLM_WORKER_MULTIPROC_METHOD=spawn PYTHONDONTWRITEBYTECODE=1
OUT=$OUTPUT_DIR/gen python scam/gen_notif.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
python -c "import pandas as pd,sys; d=pd.read_json('$OUTPUT_DIR/gen/notif_$SHARD.jsonl',lines=True); d['src']='notif'; d.to_parquet('$OUTPUT_DIR/notif_$SHARD.parquet'); print(len(d), d.intended.value_counts().to_dict())"
LABEL_IN=$OUTPUT_DIR/notif_$SHARD.parquet LABEL_OUT=$OUTPUT_DIR/labelled_$SHARD.parquet python scam/teacher_scam.py - - 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/labelled_$SHARD.parquet ] || exit 5
