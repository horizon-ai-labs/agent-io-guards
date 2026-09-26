#!/bin/bash
# usage: SRC=<model dir> BASE=<hub id> bash safety/run_release_build.sh  (release dir + ONNX + gather-int8 check on safety evals)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python ground/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
python train/export_onnx.py $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
MULTILABEL=1 GATHER_ONLY=1 python train/quant_sweep.py $OUTPUT_DIR/model $WORKSPACE/safety/evals 2>&1 | grep --line-buffered -v -E "pthread|WARNING" | tee $OUTPUT_DIR/logs/quant.log
rm -f $OUTPUT_DIR/model/onnx/model_fp16.onnx
python safety/cat_thresholds.py $OUTPUT_DIR/model $JOBS/e09a505e-38f4-4cec-865a-2cf058d620ba/safety_v0 $OUTPUT_DIR/model/thresholds.json 2>&1 | tee $OUTPUT_DIR/logs/thresholds.log
