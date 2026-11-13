#!/bin/bash
# usage: ASR_JOB=<fleurs_asr job id> TC="<truecase release job ids>" PU="<punctuation release job ids>" bash speech/run_eval_asr_text.sh -> $OUTPUT_DIR/asr_text_eval.json
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); S=""
for j in $TC; do S="$S joint:$R/$j/model"; done
for j in $PU; do S="$S punct:$R/$j/model"; done
python3 -u speech/eval_asr_text.py $OUTPUT_DIR/asr_text_eval.json $R/$ASR_JOB/fleurs_asr.parquet $S 2>&1 | grep --line-buffered -v Warn | tee $OUTPUT_DIR/logs/eval.log
