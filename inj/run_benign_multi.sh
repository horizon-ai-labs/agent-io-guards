#!/bin/bash
# usage: JOBS="<release build job id> ..." bash inj/run_benign_multi.sh -> $OUTPUT_DIR/{fpr.json,flagged.jsonl,benign_multi.parquet}
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); MODELS=""; for j in $JOBS; do MODELS="$MODELS $R/$j/model"; done
python3 -u inj/benign_multi_fpr.py $OUTPUT_DIR $MODELS 2>&1 | grep --line-buffered -v Warn | tee $OUTPUT_DIR/logs/fpr.log
