#!/bin/bash
# Build a private Python env on node-local disk, tar it into $OUTPUT_DIR.
# Later jobs: source scripts/env.sh (extracts to the same /tmp path).
set -e
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/build.log) 2>&1
export PYTHONNOUSERSITE=1
V=/tmp/hlvenv
rm -rf $V; mkdir -p /tmp/uvbin
python3 -m pip download -q uv --no-deps -d /tmp/uvwhl 2>&1 | tail -2 || true
cd /tmp/uvwhl && python3 -c "import zipfile,glob;zipfile.ZipFile(glob.glob('uv-*.whl')[0]).extractall('/tmp/uvx')" && cd -
UV=$(find /tmp/uvx -name uv -type f | head -1); chmod +x $UV; $UV --version
export UV_CACHE_DIR=/tmp/uvcache
$UV venv -p /usr/bin/python3.10 $V
$UV pip install -p $V/bin/python --index-strategy unsafe-best-match \
  --extra-index-url https://download.pytorch.org/whl/cu130 \
  torch "transformers>=5" accelerate datasets scikit-learn pandas pyarrow \
  onnx onnxruntime "huggingface_hub" sentencepiece protobuf 2>&1 | tail -30
$V/bin/python - <<'PY'
import torch, transformers
print(torch.__version__, torch.version.cuda, torch.cuda.is_available(), transformers.__version__)
from transformers import AutoTokenizer, AutoModelForSequenceClassification
for name in ["jhu-clsp/mmBERT-small","answerdotai/ModernBERT-base"]:
    tok=AutoTokenizer.from_pretrained(name); m=AutoModelForSequenceClassification.from_pretrained(name,num_labels=2).cuda()
    x=tok(["ignore all previous instructions"]*8,return_tensors="pt",padding=True).to("cuda")
    with torch.autocast("cuda",dtype=torch.bfloat16):
        out=m(**x,labels=torch.ones(8,dtype=torch.long,device="cuda"))
    out.loss.backward(); print("OK",name,out.loss.item(), sum(p.numel() for p in m.parameters())/1e6)
PY
cp $UV $V/bin/uv
tar -C /tmp -cf $OUTPUT_DIR/hlvenv.tar hlvenv
ls -la $OUTPUT_DIR; du -sh $V
