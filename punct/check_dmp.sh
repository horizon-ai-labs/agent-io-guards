#!/bin/bash
# usage: bash punct/check_dmp.sh MODEL_DIR   (private /tmp venv: transformers 4.57 + CPU torch + deepmultilingualpunctuation)
set -eo pipefail
export PYTHONNOUSERSITE=1
python3 -m venv /tmp/dmpenv && /tmp/dmpenv/bin/pip install -q --upgrade pip
/tmp/dmpenv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
/tmp/dmpenv/bin/pip install -q "transformers==4.57.*" deepmultilingualpunctuation==1.0.1 2>&1 | tail -2
/tmp/dmpenv/bin/python - "$1" <<'PY'
import sys, transformers
from deepmultilingualpunctuation import PunctuationModel
m = PunctuationModel(model=sys.argv[1])
print("transformers", transformers.__version__)
for t in ["hello how are you today i am fine thanks and you", "das ist ein test ich hoffe er funktioniert gut was meinst du",
          "my name is anna i live in berlin and i work as a nurse " * 30]:
    print("deepmultilingualpunctuation |", m.restore_punctuation(t)[:300])
PY
