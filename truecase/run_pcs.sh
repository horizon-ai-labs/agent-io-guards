#!/bin/bash
# CPU job. usage: bash truecase/run_pcs.sh -> $OUTPUT_DIR/pcs_preds.json (private venv in /tmp: punctuators brings its own torch)
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
export PYTHONNOUSERSITE=1
python3 -m venv /tmp/pcs && /tmp/pcs/bin/pip install -q --upgrade pip
/tmp/pcs/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/tmp/pcs/bin/pip install -q punctuators pandas pyarrow 2>&1 | tail -3
/tmp/pcs/bin/python truecase/pcs_preds.py truecase/evals $OUTPUT_DIR/pcs_preds.json 2>&1 | tee $OUTPUT_DIR/logs/pcs.log
