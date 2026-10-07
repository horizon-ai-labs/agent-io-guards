#!/bin/bash
# usage: DATA=<dir with train/val> EVALS=<eval dir> BASE=jhu-clsp/mmBERT-small LR=5e-5 SEED=0 EPOCHS=2 EXTRA='--lower 0.7' bash punct/run_train3.sh
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
hostname | tee $OUTPUT_DIR/logs/host.log
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python3 punct/train_punct.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model --lr ${LR:-5e-5} \
  --seed ${SEED:-0} --epochs ${EPOCHS:-1} ${EXTRA} 2>&1 | tee $OUTPUT_DIR/logs/train.log
python3 punct/eval_punct.py $OUTPUT_DIR/eval_results.json $EVALS $OUTPUT_DIR/model 2>&1 | tee $OUTPUT_DIR/logs/eval.log
