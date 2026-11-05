#!/bin/bash
# (never name an env var CC: it is the C compiler for torch inductor / triton)
# usage: [REUSE=<job-id>] SYSTEMS="none random <model dir or hub id> ..." [RATES=0.5,0.33] READER_DIR=<job-id>/model bash compress/run_eval_qa.sh
#   -> $OUTPUT_DIR/compressed.parquet, $OUTPUT_DIR/qa_result.json (+ _rows.parquet). Model dirs may be given as <job-id> (-> <job-id>/model).
set -o pipefail; mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/eval.log) 2>&1
python3 -c "import torch; assert torch.cuda.is_available(); torch.zeros(1).cuda()" || { echo "GPU unusable on $(hostname)"; exit 4; }
R=$(dirname $OUTPUT_DIR); S=""
for s in $SYSTEMS; do if [ -d "$R/$s/model" ]; then S="$S $R/$s/model"; else S="$S $s"; fi; done
if [ -n "$REUSE" ]; then cp $R/$REUSE/compressed.parquet $OUTPUT_DIR/compressed.parquet || exit 6   # REUSE=<job-id>: stage A already done there
else python3 compress/compress_ctx.py compress/evals/qa.parquet $OUTPUT_DIR/compressed.parquet ${RATES:-0.5,0.33} $S 2>&1 | grep --line-buffered -v Warn || exit 6; fi
VLLM_TAR=$R/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
export MODEL=$R/$READER_DIR HF_HUB_OFFLINE=1 QUANT=${QUANT:-fp8}
PYTHONDONTWRITEBYTECODE=1 VLLM_WORKER_MULTIPROC_METHOD=spawn QA=compress/evals/qa.parquet CTX_FILE=$OUTPUT_DIR/compressed.parquet OUT=$OUTPUT_DIR/qa_result.json \
  python compress/read_qa.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
[ -s $OUTPUT_DIR/qa_result.json ] || exit 5
