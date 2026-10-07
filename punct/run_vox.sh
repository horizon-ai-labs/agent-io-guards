#!/bin/bash
# CPU job: VoxPopuli windows + merged training dir (v1 data + vox) -> $OUTPUT_DIR/data_vox/{train,val}.parquet
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
PREP=$JOBS/2cd83945-7782-40e5-b739-31df2550930d
python3 punct/build_vox.py $OUTPUT_DIR $PREP/evals 15000 2>&1 | tee $OUTPUT_DIR/logs/vox.log
mkdir -p $OUTPUT_DIR/data_vox
python3 -c "
import pandas as pd
t=pd.read_parquet('$PREP/data/train.parquet'); v=pd.read_parquet('$OUTPUT_DIR/vox.parquet')
pd.concat([t,v],ignore_index=True).sample(frac=1.0,random_state=0).to_parquet('$OUTPUT_DIR/data_vox/train.parquet')
import shutil; shutil.copy('$PREP/data/val.parquet','$OUTPUT_DIR/data_vox/val.parquet'); print(len(t),len(v))
" 2>&1 | tee -a $OUTPUT_DIR/logs/vox.log
