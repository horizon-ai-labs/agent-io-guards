#!/bin/bash
# usage: [RESUME=<job-id>] bash speech/run_fleurs_asr.sh -> $OUTPUT_DIR/fleurs_asr.parquet
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
export HF_HOME=${TMPDIR:-/tmp}/hf PYTHONDONTWRITEBYTECODE=1
[ -n "$RESUME" ] && cp $(dirname $OUTPUT_DIR)/$RESUME/fleurs_asr.parquet $OUTPUT_DIR/fleurs_asr.parquet
python3 -u speech/fleurs_asr.py $OUTPUT_DIR/fleurs_asr.parquet 2>&1 | grep --line-buffered -v -E "Warn|it/s\]" | tee $OUTPUT_DIR/logs/asr.log
