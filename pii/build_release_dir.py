"""Assemble a PII release dir: weights + base tokenizer_config (transformers-4.x compatible) + redact.py.
python pii/build_release_dir.py SRC BASE_ID OUT"""
import os, shutil, sys
from huggingface_hub import hf_hub_download
src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "val_metrics.json", "train_log.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release", "pii", "redact.py"), f"{out}/redact.py")
sys.path.insert(0, out)
from redact import PIIRedactor
print(PIIRedactor(out).redact("Hi, I'm Anna Müller, mail anna.mueller@posteo.de, key sk-proj-9fQ2x7LmA1bC3dE4fG5hI6jK"))
