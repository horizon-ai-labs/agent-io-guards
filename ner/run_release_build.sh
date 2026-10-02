#!/bin/bash
# usage: SRC=<trained model dir> BASE=<hub id> bash ner/run_release_build.sh  (release dir + ONNX + int8 token agreement + pipeline smoke test)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python - <<PY 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/build.log
import os, shutil
from huggingface_hub import hf_hub_download
src, out = "$SRC", "$OUTPUT_DIR/model"; os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "val_metrics.json", "train_log.json"]:
    if os.path.exists(f"{src}/{f}"): shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]: shutil.copy(hf_hub_download("$BASE", f), f"{out}/{f}")
from transformers import pipeline
ner = pipeline("token-classification", model=out, aggregation_strategy="simple", device=0)
for t in ["Angela Merkel met Emmanuel Macron in Paris to discuss the European Union.", "El Real Madrid fichó al delantero brasileño Vinícius.", "李华在北京大学学习中文。"]:
    print(t, [(e["word"], e["entity_group"], round(float(e["score"]), 2)) for e in ner(t)])
PY
python pii/export_onnx_tok.py $OUTPUT_DIR/model $WORKSPACE/.gated/ner/v0/val.parquet 2>&1 | grep --line-buffered -v -E "Warning|pthread" | tee $OUTPUT_DIR/logs/export.log
ls -la $OUTPUT_DIR/model $OUTPUT_DIR/model/onnx
