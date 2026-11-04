#!/bin/bash
# usage: DATA_JOB=<punct data job id> JOBS="<release build job id> ..." bash truecase/run_boundary.sh -> $OUTPUT_DIR/boundary.json
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); MODELS=""; for j in $JOBS; do MODELS="$MODELS $R/$j/model"; done
python3 truecase/boundary_check.py $OUTPUT_DIR/boundary.json $R/$DATA_JOB/data truecase/evals $MODELS 2>&1 | grep -v Warn | tee $OUTPUT_DIR/logs/boundary.log
