#!/bin/bash
# usage: PREP=<prep job output dir> bash punct/run_eval_baselines.sh
set -e; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 punct/eval_punct.py $OUTPUT_DIR/eval_results.json $PREP/evals kredor/punctuate-all oliverguhr/fullstop-punctuation-multilang-large \
  oliverguhr/fullstop-punctuation-multilingual-base 2>&1 | tee $OUTPUT_DIR/logs/eval.log
