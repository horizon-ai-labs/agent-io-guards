#!/bin/bash
# usage: DOCS_JOB=<job with fw/test_docs.parquet> JOBS="<release build job ids>" bash inj/run_longdoc_fpr.sh -> $OUTPUT_DIR/longdoc_fpr.json
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); M=""; for j in $JOBS; do M="$M $R/$j/model"; done
python3 -u inj/longdoc_fpr.py $OUTPUT_DIR $R/$DOCS_JOB/fw/test_docs.parquet $M 2>&1 | grep --line-buffered -v Warn | tee $OUTPUT_DIR/logs/fpr.log
