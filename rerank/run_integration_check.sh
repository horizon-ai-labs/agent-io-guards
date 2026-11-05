#!/bin/bash
# CPU job. usage: JOB=<release build job id> bash rerank/run_integration_check.sh -> $OUTPUT_DIR/integrations.json (private venv in /tmp:
# the frameworks bring their own torch/transformers)
set -o pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
export PYTHONNOUSERSITE=1
python3 -m venv /tmp/iv && /tmp/iv/bin/pip install -q --upgrade pip
/tmp/iv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2
/tmp/iv/bin/pip install -q sentence-transformers langchain langchain-community langchain-classic llama-index-core haystack-ai sentence-transformers-haystack 2>&1 | tail -3
/tmp/iv/bin/python rerank/integration_check.py $(dirname $OUTPUT_DIR)/$JOB/model 2>&1 | grep -v -E "Warning|warn" | tee $OUTPUT_DIR/integrations.json
