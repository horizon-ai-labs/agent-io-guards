#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub id> EV=<eval dir> bash lid/run_release_build.sh  (release dir + ONNX + gather-int8 argmax check)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python ground/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
for d in lid_v5 lid_v3; do [ -f $SRC/../$d/labels.json ] && { cp $SRC/../$d/labels.json $OUTPUT_DIR/model/labels.json; break; }; done
python train/export_onnx.py $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
MULTICLASS=1 QN=200 QMAX=400 GATHER_ONLY=1 MIN_AGREE=0.99 python train/quant_sweep.py $OUTPUT_DIR/model $EV 2>&1 | grep --line-buffered -v -E "pthread|WARNING" | tee $OUTPUT_DIR/logs/quant.log
rm -f $OUTPUT_DIR/model/onnx/model_fp16.onnx
