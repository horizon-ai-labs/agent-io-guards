#!/bin/bash
# usage: MODELS="spec,spec" [EXTRA="--max_len 128"] bash edu/run_eval.sh -> $OUTPUT_DIR/eval.json  (spec: edu/eval_edu.py; local job outputs as @<job-id>/<subdir>)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
export PYTHONDONTWRITEBYTECODE=1
M=$(echo "$MODELS" | sed "s#@#$(dirname $OUTPUT_DIR)/#g")
python edu/eval_edu.py --models "$M" --evals ${EVALS:-edu/evals} --out $OUTPUT_DIR/eval.json $EXTRA 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
