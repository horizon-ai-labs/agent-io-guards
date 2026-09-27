#!/bin/bash
# usage: ITEMS=<json> SHARD=i NSHARD=n bash safety/run_translate.sh -> $OUTPUT_DIR/tr/tr_<i>.jsonl   (ITEMS=prep: build items first)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/translate.log) 2>&1
R=$JOBS
if [ "$ITEMS" = prep ]; then
  ( source scripts/env.sh; export HF_HOME=${TMPDIR:-/tmp}/hf; python safety/translate_prep.py $R/691f6b5c-a835-4c5d-8cd3-d73598352cd4/pool_labelled.parquet $R/d4120ea3-5411-49eb-b77b-f2104691421d/labelled.parquet $OUTPUT_DIR/items.json ) || exit 1
  ITEMS=$OUTPUT_DIR/items.json
fi
VLLM_TAR=$R/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
ITEMS=$ITEMS OUT=$OUTPUT_DIR/tr python safety/translate_items.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
