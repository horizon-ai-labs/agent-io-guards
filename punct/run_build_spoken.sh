#!/bin/bash
# CPU job. usage: GEN="dir1 dir2 ..." bash punct/run_build_spoken.sh -> $OUTPUT_DIR/{spoken.parquet,data/}
set -eo pipefail; cd $WORKSPACE; mkdir -p $OUTPUT_DIR/logs
PREP=$JOBS/2cd83945-7782-40e5-b739-31df2550930d
FILES=$(for d in $GEN; do ls $d/gen/spoken_*.jsonl; done)
python3 punct/build_spoken.py $OUTPUT_DIR $PREP/evals $PREP/data $FILES 2>&1 | tee $OUTPUT_DIR/logs/build.log
