#!/bin/bash
# usage: BASE=... EXTRA="..." bash safety/run_v4_all.sh  (teacher-label the translations of job a95aa959, then v4 build/train/eval)
set -eo pipefail
R=$JOBS
OUT0=$OUTPUT_DIR
TR=$R/a95aa959-cae5-41dd-aee8-bf49be6530a2/tr OUTPUT_DIR=$OUT0/label bash safety/run_label_tr.sh
TRL=$OUT0/label/labelled.parquet OUTPUT_DIR=$OUT0 bash safety/run_train_v4.sh
