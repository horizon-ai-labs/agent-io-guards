#!/bin/bash
# usage: TR=<dir with tr_*.jsonl> bash safety/run_label_tr.sh -> $OUTPUT_DIR/labelled.parquet (all translated items, teacher-scored)
set -eo pipefail
source scripts/env.sh
mkdir -p $OUTPUT_DIR/logs
python -c "
import pandas as pd, glob, json
rows = [json.loads(l) for p in sorted(glob.glob('$TR/tr_*.jsonl')) for l in open(p)]
df = pd.DataFrame(rows); df['text_pair'] = ''; df['kind'] = 'prompt'
df.to_parquet('$OUTPUT_DIR/tr.parquet'); print(len(df), df.groupby('lang').size().to_dict())
"
python safety/label_teacher.py $OUTPUT_DIR/tr.parquet $OUTPUT_DIR/labelled.parquet ${TEACHER:-Qwen/Qwen3Guard-Gen-8B} 2>&1 | grep --line-buffered -v Warning | tee $OUTPUT_DIR/logs/label.log
