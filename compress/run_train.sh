#!/bin/bash
# usage: BASE=jhu-clsp/mmBERT-small LR=5e-5 SEED=0 EPOCHS=2 bash compress/run_train.sh -> $OUTPUT_DIR/model
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python3 compress/train_comp.py --base ${BASE:-jhu-clsp/mmBERT-small} --train .gated/compress/data/train.parquet --val .gated/compress/data/val.parquet \
  --out $OUTPUT_DIR/model --lr ${LR:-5e-5} --seed ${SEED:-0} --epochs ${EPOCHS:-2} 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
