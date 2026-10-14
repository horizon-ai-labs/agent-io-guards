#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub id> bash question/run_release_build_kw.sh  (release dir + ONNX + int8 all-label check)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python ground/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
python train/export_onnx.py $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
QN=400 QMAX=400 GATHER_ONLY=1 MIN_AGREE=0.97 python train/quant_sweep.py $OUTPUT_DIR/model question/evals_kw 2>&1 | grep --line-buffered -v -E "pthread|WARNING" | tee $OUTPUT_DIR/logs/quant.log
rm -f $OUTPUT_DIR/model/onnx/model_fp16.onnx
ls -la $OUTPUT_DIR/model $OUTPUT_DIR/model/onnx
