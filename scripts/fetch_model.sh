#!/bin/bash
# usage: REPO=Qwen/Qwen3.8-27B-FP8 bash scripts/fetch_model.sh -> $OUTPUT_DIR/model (one copy on job storage; later jobs use MODEL_DIR=<job-id>/model
#        instead of downloading from the Hub each time, which hits unauthenticated rate limits when several jobs start together).
# Downloads to node-local /tmp first (Hub downloads straight onto the shared filesystem stall on file locks), then copies.
set -o pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs; L=$OUTPUT_DIR/logs/fetch.log
export HF_HOME=${TMPDIR:-/tmp}/hf; D=${TMPDIR:-/tmp}/dl
for k in 1 2 3 4 5 6; do
  python -c "from huggingface_hub import snapshot_download as s; print(s('$REPO', local_dir='$D', max_workers=8))" >> $L 2>&1 && break
  echo "retry $k $(date)" >> $L; tail -n 2 $L; sleep $((60 * k))
done
du -sh $D >> $L; cp -r $D $OUTPUT_DIR/model && rm -rf $OUTPUT_DIR/model/.cache; du -sh $OUTPUT_DIR/model >> $L; ls $OUTPUT_DIR/model >> $L
