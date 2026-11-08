#!/bin/bash
# CPU job. usage: JOB=<injection release build job id> bash inj/run_llmguard_check.sh -> $OUTPUT_DIR/llmguard.json (private venv: llm-guard pins its own stack)
set -o pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
export PYTHONNOUSERSITE=1
python3 -m venv /tmp/lg && /tmp/lg/bin/pip install -q --upgrade pip
/tmp/lg/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu 2>&1 | tail -2
/tmp/lg/bin/pip install -q "llm-guard[onnxruntime]" 2>&1 | tail -3
/tmp/lg/bin/python inj/llmguard_check.py $(dirname $OUTPUT_DIR)/$JOB/model 2>&1 | grep -v -E "Warning|warn" | tee $OUTPUT_DIR/llmguard.json
