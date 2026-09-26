#!/bin/bash
# usage: bash safety/run_build.sh -> $OUTPUT_DIR/safety_v0
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
export HF_HOME=${TMPDIR:-/tmp}/hf
python safety/build_safety_v0.py $OUTPUT_DIR/safety_v0 2>&1 | tee $OUTPUT_DIR/logs/build.log
