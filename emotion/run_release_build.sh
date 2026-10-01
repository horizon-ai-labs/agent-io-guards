#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub id> bash emotion/run_release_build.sh  (release dir + thresholds + ONNX + int8 check)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python ground/build_release_dir.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
python - <<PY
import json, sys
sys.path.insert(0, "emotion")
from common import EKMAN
r = list(json.load(open("$SRC/../eval.json")).values())[0]
json.dump({k: round(v, 3) for k, v in r["goemo"]["thresholds"].items()}, open("$OUTPUT_DIR/model/thresholds.json", "w"), indent=1)
json.dump(EKMAN, open("$OUTPUT_DIR/model/ekman_mapping.json", "w"), indent=1)
PY
python train/export_onnx.py $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
MULTILABEL=all QN=8 QMAX=448 GATHER_ONLY=1 MIN_AGREE=0.97 python train/quant_sweep.py $OUTPUT_DIR/model emotion/evals/brighter 2>&1 | grep --line-buffered -v -E "pthread|WARNING" | tee $OUTPUT_DIR/logs/quant.log
rm -f $OUTPUT_DIR/model/onnx/model_fp16.onnx
ls -la $OUTPUT_DIR/model $OUTPUT_DIR/model/onnx
