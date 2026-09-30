#!/bin/bash
# usage: PER=6000 bash lid/run_prep.sh -> $OUTPUT_DIR/lid_v0 (CPU job)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
export HF_HOME=${TMPDIR:-/tmp}/hf HF_DATASETS_CACHE=${TMPDIR:-/tmp}/hf/ds
python lid/prep_lid.py $OUTPUT_DIR/lid_v0 ${PER:-6000} 2>&1 | grep --line-buffered -vE "it/s\]|%\|" | tee $OUTPUT_DIR/logs/prep.log
