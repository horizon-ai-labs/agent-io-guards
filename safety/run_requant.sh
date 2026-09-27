#!/bin/bash
# usage: SRC=<release model dir with onnx/model.onnx> MIN_AGREE=0.99 bash safety/run_requant.sh -> $OUTPUT_DIR/model (copy + gather-int8)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs $OUTPUT_DIR/model/onnx
cp $SRC/*.json $SRC/model.safetensors $OUTPUT_DIR/model/ && cp $SRC/onnx/model.onnx $OUTPUT_DIR/model/onnx/
MIN_AGREE=${MIN_AGREE:-0.995} MULTILABEL=1 GATHER_ONLY=1 python train/quant_sweep.py $OUTPUT_DIR/model $WORKSPACE/safety/evals 2>&1 | grep --line-buffered -v -E "pthread|WARNING" | tee $OUTPUT_DIR/logs/quant.log
rm -f $OUTPUT_DIR/model/onnx/model_fp16.onnx
ls -la $OUTPUT_DIR/model/onnx
