#!/bin/bash
# usage: DATA=<punct data dir with train/val> BASE=jhu-clsp/mmBERT-small LR=5e-5 SEED=0 EPOCHS=2 EXTRA='' bash truecase/run_train.sh
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
python3 truecase/train_tc.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model --lr ${LR:-5e-5} \
  --seed ${SEED:-0} --epochs ${EPOCHS:-1} ${EXTRA} 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python3 truecase/eval_tc.py $OUTPUT_DIR/eval.json truecase/evals joint:$OUTPUT_DIR/model 2>&1 | grep -v Warn | tee $OUTPUT_DIR/logs/eval.log
