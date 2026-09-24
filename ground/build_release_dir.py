"""Groundedness release dir: weights + base tokenizer_config (transformers-4.x compatible). python ground/build_release_dir.py SRC BASE OUT"""
import os, shutil, sys
from huggingface_hub import hf_hub_download
src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "val_metrics.json", "train_log.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
from transformers import pipeline
clf = pipeline("text-classification", model=out, device=0)
print(clf({"text": "The Eiffel Tower is 330 metres tall and was completed in 1889.", "text_pair": "The Eiffel Tower was completed in 1889."}),
      clf({"text": "The Eiffel Tower is 330 metres tall and was completed in 1889.", "text_pair": "The Eiffel Tower was completed in 1899."}))
