#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub base id> bash ncii/run_release_build.sh  (release dir + ONNX + int8 sweep)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python ncii/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep -v Warning | tee $OUTPUT_DIR/logs/build.log
python train/export_onnx.py $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
QN=400 python train/quant_sweep.py $OUTPUT_DIR/model .gated/ncii/qev 2>&1 | grep --line-buffered -v -E "pthread|WARNING" | tee $OUTPUT_DIR/logs/quant.log
rm -f $OUTPUT_DIR/model/onnx/model_fp16.onnx
