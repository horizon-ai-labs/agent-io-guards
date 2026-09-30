#!/bin/bash
# usage: MODELS=a,b EVALS=<dir> bash lid/run_eval.sh  (fastText baselines need a private venv with fasttext)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
if echo "$MODELS" | grep -q "fasttext:"; then
  python3 -m venv --system-site-packages /tmp/ftv && /tmp/ftv/bin/pip install -q fasttext-wheel 2>&1 | tail -2
  PY=/tmp/ftv/bin/python
else PY=python; fi
$PY lid/eval_lid.py --models $MODELS --evals $EVALS --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
