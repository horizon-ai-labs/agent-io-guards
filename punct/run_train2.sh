#!/bin/bash
# usage: PREP=<prep job output dir> BASE=jhu-clsp/mmBERT-small LR=5e-5 SEED=0 bash punct/run_train2.sh
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
hostname | tee $OUTPUT_DIR/logs/host.log
python3 -c 'import torch,sys; sys.exit(0 if torch.cuda.is_available() else 4)' || { echo "no CUDA on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
nvidia-smi --query-gpu=name,memory.total --format=csv | tee $OUTPUT_DIR/logs/gpu.log
python3 punct/train_punct.py --base ${BASE:-jhu-clsp/mmBERT-small} --data $PREP/data --out $OUTPUT_DIR/model --lr ${LR:-5e-5} \
  --seed ${SEED:-0} --epochs ${EPOCHS:-1} ${EXTRA} 2>&1 | tee $OUTPUT_DIR/logs/train.log
python3 punct/eval_punct.py $OUTPUT_DIR/eval_results.json $PREP/evals $OUTPUT_DIR/model 2>&1 | tee $OUTPUT_DIR/logs/eval.log
