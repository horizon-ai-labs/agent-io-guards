#!/bin/bash
# usage: N=150 bash safety/run_edgy.sh -> $OUTPUT_DIR/edgy/edgy_0.jsonl, then teacher labels -> $OUTPUT_DIR/labelled.parquet
mkdir -p $OUTPUT_DIR/logs
R=$JOBS
( VLLM_TAR=$R/e5246a75-1a48-4e1c-a74f-e8a519648222/vllmenv.tar source gen/vllm_env.sh
  N=${N:-150} OUT=$OUTPUT_DIR/edgy python safety/gen_edgy.py 2>&1 | grep --line-buffered -vE "it/s\]|%\||INFO|WARNING" | tee $OUTPUT_DIR/logs/gen.log )
source scripts/env.sh
python -c "
import pandas as pd, json
rows = [json.loads(l) for l in open('$OUTPUT_DIR/edgy/edgy_0.jsonl')]
df = pd.DataFrame(rows).drop_duplicates(subset=['text']); df['text_pair'] = ''; df['kind'] = 'prompt'; df['source'] = 'edgy/' + df.intended
df.to_parquet('$OUTPUT_DIR/edgy.parquet'); print(len(df), df.intended.value_counts().to_dict())
"
python safety/label_teacher.py $OUTPUT_DIR/edgy.parquet $OUTPUT_DIR/labelled.parquet Qwen/Qwen3Guard-Gen-8B 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/label.log
