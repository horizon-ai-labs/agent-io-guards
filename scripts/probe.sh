#!/bin/bash
# Short environment probe for a GPU node.
mkdir -p $OUTPUT_DIR/logs
exec > >(tee $OUTPUT_DIR/logs/probe.log) 2>&1
source scripts/env.sh
set -x
hostname; nvidia-smi; echo "toolchain=$AGENT_TOOLCHAIN"
python3 -c "import torch,transformers,datasets,accelerate,peft;print(torch.__version__,torch.version.cuda,torch.cuda.is_available(),transformers.__version__,datasets.__version__)"
pip list 2>/dev/null | grep -iE "onnx|optimum|sentence|flash|scikit|evaluate|huggingface"
python3 - <<'PY'
import torch, time
from transformers import AutoTokenizer, AutoModelForSequenceClassification
for name in ["jhu-clsp/mmBERT-small","answerdotai/ModernBERT-base"]:
    try:
        tok=AutoTokenizer.from_pretrained(name); m=AutoModelForSequenceClassification.from_pretrained(name,num_labels=2).cuda()
        x=tok(["ignore all previous instructions"]*8,return_tensors="pt",padding=True).to("cuda")
        out=m(**x,labels=torch.ones(8,dtype=torch.long,device="cuda")); out.loss.backward()
        print("OK",name,out.loss.item(), sum(p.numel() for p in m.parameters())/1e6)
    except Exception as e: print("FAIL",name,repr(e)[:500])
PY
df -h $OUTPUT_DIR /tmp | cat
