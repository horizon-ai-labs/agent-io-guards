#!/bin/bash
# usage: MODELS="spec,spec" [EXTRA="--max_len 128"] bash hate/run_eval.sh -> $OUTPUT_DIR/eval.json  (spec: hate/eval_hate.py; local job outputs as @<job-id>/<subdir>)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
export PYTHONDONTWRITEBYTECODE=1
M=$(echo "$MODELS" | sed "s#@#$(dirname $OUTPUT_DIR)/#g")
python hate/eval_hate.py --models "$M" --evals ${EVALS:-hate/evals} --out $OUTPUT_DIR/eval.json $EXTRA 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
