#!/bin/bash
# usage: MODELS="<job-id or hub-id> ..." bash ncii/run_eval.sh -> $OUTPUT_DIR/eval.json  (<job-id> -> <job-id>/model; else used as-is)
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); M=""
for m in $MODELS; do if [ -d "$R/$m/model" ]; then M="$M $R/$m/model"; else M="$M $m"; fi; done
python3 ncii/eval_ncii.py $OUTPUT_DIR/eval.json .gated/ncii/data/test.parquet $M 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/eval.log
