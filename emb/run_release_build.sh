#!/bin/bash
# usage: SRC=<student dir> BASE=<hub id> bash emb/run_release_build.sh -> $OUTPUT_DIR/model (sentence-transformers layout + ONNX)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python emb/build_release.py $SRC $BASE $OUTPUT_DIR/model 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/build.log
python -c "import sentence_transformers" 2>/dev/null && python - <<PY 2>&1 | tee -a $OUTPUT_DIR/logs/build.log
from sentence_transformers import SentenceTransformer
m = SentenceTransformer("$OUTPUT_DIR/model", device="cuda")
e = m.encode(["How tall is the Eiffel Tower?", "La tour Eiffel mesure 330 mètres.", "The cat sat on the mat."])
print("st shape", e.shape, "sims", (e @ e.T).round(3).tolist())
PY
ls -la $OUTPUT_DIR/model $OUTPUT_DIR/model/onnx
