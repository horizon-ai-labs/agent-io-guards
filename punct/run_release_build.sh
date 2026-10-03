#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub id> PREP=<prep dir> bash punct/run_release_build.sh  (release dir + ONNX + int8 check)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
pip install --no-deps --target /tmp/dmp deepmultilingualpunctuation==1.0.1 >/dev/null 2>&1 && export PYTHONPATH=/tmp/dmp
python punct/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
bash punct/check_dmp.sh $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/dmp.log || echo "dmp check failed"
python pii/export_onnx_tok.py $OUTPUT_DIR/model $PREP/evals/ted.parquet 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
ls -la $OUTPUT_DIR/model $OUTPUT_DIR/model/onnx
