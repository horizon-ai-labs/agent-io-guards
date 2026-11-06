#!/bin/bash
# usage: BASE=jhu-clsp/mmBERT-small LR=5e-5 SEED=0 EPOCHS=2 bash ncii/run_train.sh -> $OUTPUT_DIR/model (+ obfuscation normalizer)
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python3 train/train.py --base ${BASE:-jhu-clsp/mmBERT-small} --data .gated/ncii/data --out $OUTPUT_DIR/model --labels SAFE,NCII \
  --normalizer --lr ${LR:-5e-5} --seed ${SEED:-0} --epochs ${EPOCHS:-2} --max_len 256 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
