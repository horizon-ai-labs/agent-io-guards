"""Scam-detector release dir: weights + base tokenizer_config (transformers-4.x compatible).  python scam/build_release_dir.py SRC BASE OUT"""
import os, shutil, sys
from huggingface_hub import hf_hub_download
src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "train_log.json", "val_metrics.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
from transformers import pipeline
clf = pipeline("text-classification", model=out, device=0, top_k=None)
for t in ["Your parcel could not be delivered. Pay the 1.99 EUR customs fee here: dhl-redelivery.example.com",
          "Hi, running 10 min late, see you at the cafe", "Your verification code is 482913. Do not share it with anyone.",
          "Ihr Konto wurde gesperrt. Bestätigen Sie Ihre Daten innerhalb von 24 Stunden: sparkasse-sicher.example.de",
          "MEGA SALE 70% off all shoes this weekend only! Shop now at shoeworld.example.com"]:
    r = clf(t); r = r[0] if r and isinstance(r[0], list) else r
    print({d["label"]: round(d["score"], 3) for d in r}, "|", t[:70], flush=True)
