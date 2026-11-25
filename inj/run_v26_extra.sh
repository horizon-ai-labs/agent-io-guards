#!/bin/bash
# usage: REL_JOB=<release build job id> bash inj/run_v26_extra.sh -> third-party tr/ja evals (inj/evals_new) + Aya/OASST2 benign false
# alarms (inj/benign_multi_fpr.py) for a release dir, so the card's sections can be refreshed for the new version.
set -eo pipefail; cd $WORKSPACE; source scripts/env.sh; mkdir -p $OUTPUT_DIR/logs
python -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); M=$R/$REL_JOB/model
python train/evaluate.py --normalize --evals $WORKSPACE/inj/evals_new --models $M --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/eval.log
mkdir -p $OUTPUT_DIR/benign; python -u inj/benign_multi_fpr.py $OUTPUT_DIR/benign $M 2>&1 | grep --line-buffered -v Warn | tee $OUTPUT_DIR/logs/fpr.log
