#!/bin/bash
# usage: bash sentiment/run_eval0.sh -> $OUTPUT_DIR/baselines.json (transformers baselines), $OUTPUT_DIR/teacher.json (Qwen3.8-27B)
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/eval0.log) 2>&1
source scripts/env.sh
python sentiment/eval_sent.py --evals sentiment/evals --out $OUTPUT_DIR/baselines.json --models \
  cardiffnlp/twitter-xlm-roberta-base-sentiment,lxyuan/distilbert-base-multilingual-cased-sentiments-student,nlptown/bert-base-multilingual-uncased-sentiment,tabularisai/multilingual-sentiment-analysis,cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual,clapAI/modernBERT-base-multilingual-sentiment \
  2>&1 | grep --line-buffered -vE "it/s\]|Warning"
VLLM_TAR=$JOBS/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
python sentiment/teacher_sent.py $OUTPUT_DIR/teacher.json sentiment/evals 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING"
