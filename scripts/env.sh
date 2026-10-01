# Source at the start of every GPU/CPU job.
# 1) Use the shared user-site stack if it is healthy (torch 2.14 + transformers 5 model classes import).
# 2) Otherwise fall back to our private venv (scripts/build_venv.sh), a tar extracted to node-local /tmp.
#    (2026-09-23: the shared site was once broken by others' installs; one node hit an I/O error reading the tar.)
export TOKENIZERS_PARALLELISM=false
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}
if python3 -c "import torch,sklearn,pandas,onnxruntime; assert torch.__version__.startswith('2.14'); from transformers import ModernBertForSequenceClassification" 2>/dev/null; then
  echo "[env] using shared stack"
  mkdir -p /tmp/hlbin; ln -sf $(command -v python3) /tmp/hlbin/python; export PATH=/tmp/hlbin:$PATH
else
  VENV_TAR=${VENV_TAR:-$JOBS/2d71b5d1-a9f2-443d-ae4c-01261c7d30b2/hlvenv.tar}
  export PYTHONNOUSERSITE=1
  for i in 1 2 3; do
    [ -x /tmp/hlvenv/bin/python ] && break
    tar -C /tmp -xf $VENV_TAR && break
    echo "[env] tar extract failed (try $i), retrying"; rm -rf /tmp/hlvenv; sleep 20
  done
  export PATH=/tmp/hlvenv/bin:$PATH
  echo "[env] using private venv"
fi
python -c "import torch,transformers;print('[env]',torch.__version__,torch.version.cuda,torch.cuda.is_available(),transformers.__version__)"
