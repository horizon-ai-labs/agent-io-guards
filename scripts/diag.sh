#!/bin/bash
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/diag.log) 2>&1
hostname; echo HOME=$HOME PYTHONUSERBASE=$PYTHONUSERBASE PYTHONPATH=$PYTHONPATH TMPDIR=$TMPDIR
which python3; python3 -m site
python3 -c "import torch;print(torch.__version__, torch.__file__)"
python3 -c "import transformers;print(transformers.__version__, transformers.__file__)"
python3 -c "import torchvision;print(torchvision.__version__, torchvision.__file__)"
ls $(python3 -m site --user-site) | head -300
pip --version
env | grep -iE "python|pip|conda|virtual" 
