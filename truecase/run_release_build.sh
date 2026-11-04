#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub id> bash truecase/run_release_build.sh  (release dir + helper check + ONNX fp32/int8 + agreement)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python truecase/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model truecase/evals 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
python pii/export_onnx_tok.py $OUTPUT_DIR/model $OUTPUT_DIR/ted_lower.parquet 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
ls -la $OUTPUT_DIR/model $OUTPUT_DIR/model/onnx
