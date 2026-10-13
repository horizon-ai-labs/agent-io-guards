#!/bin/bash
# usage: DATA=<dir> BASE=... EXTRA="..." bash prompt/run_train.sh  (tox/train_tox.py with the 17 prompt outputs, then prompt/eval_prompt.py)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "import torch; assert torch.cuda.is_available(), 'no usable GPU'" || { echo "GPU unusable on $(hostname)"; exit 4; }
export PYTHONDONTWRITEBYTECODE=1
L=task_open_qa,task_closed_qa,task_summarization,task_text_generation,task_code_generation,task_chatbot,task_classification,task_rewrite,task_brainstorming,task_extraction,task_other,creativity,reasoning,contextual_knowledge,domain_knowledge,constraints,complexity
python tox/train_tox.py --labels $L --max_len 512 --base ${BASE:-jhu-clsp/mmBERT-small} --data $DATA --out $OUTPUT_DIR/model $EXTRA 2>&1 | grep --line-buffered -v "it/s\]" | tee $OUTPUT_DIR/logs/train.log
python prompt/eval_prompt.py --models $OUTPUT_DIR/model --ref prompt/evals/test_judged.parquet --out $OUTPUT_DIR/eval.json 2>&1 | grep --line-buffered -vE "it/s\]|Warning" | tee $OUTPUT_DIR/logs/eval.log
