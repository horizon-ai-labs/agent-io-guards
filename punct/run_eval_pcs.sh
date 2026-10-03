#!/bin/bash
# CPU job. usage: PREP=<prep dir> bash punct/run_eval_pcs.sh   (private venv in /tmp: punctuators brings its own torch)
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
export PYTHONNOUSERSITE=1
python3 -m venv /tmp/pcs && /tmp/pcs/bin/pip install -q --upgrade pip
/tmp/pcs/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/tmp/pcs/bin/pip install -q punctuators pandas pyarrow 2>&1 | tail -3
/tmp/pcs/bin/python punct/eval_pcs.py $OUTPUT_DIR/eval_results.json $PREP/evals 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase pcs_47lang 2>&1 | tee $OUTPUT_DIR/logs/eval.log
