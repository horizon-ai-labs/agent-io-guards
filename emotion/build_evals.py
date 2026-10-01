"""Emotion evaluation sets (evaluation only).   python emotion/build_evals.py OUT_DIR
- goemo_{test,dev}.parquet: google-research-datasets/go_emotions "simplified" test / validation (English Reddit, 28 labels,
  Apache-2.0). Columns: text, labels (list of label names).
- brighter/<lang>_{test,dev}.parquet: brighter-dataset/BRIGHTER-emotion-categories (CC-BY-4.0; human labels for anger,
  disgust, fear, joy, sadness, surprise, multi-label; 28 languages). Test capped at 1,500 per language (random); dev kept
  whole for threshold tuning. Columns: text + 6 emotion columns.
"""
import os, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GO, E6

OUT = sys.argv[1]
os.makedirs(f"{OUT}/brighter", exist_ok=True)
for sp, name in [("test", "test"), ("validation", "dev")]:
    d = pd.read_parquet(hf_hub_download("google-research-datasets/go_emotions", f"simplified/{sp}-00000-of-00001.parquet", repo_type="dataset"))
    d = d.assign(labels=d.labels.map(lambda ls: [GO[i] for i in ls]))[["text", "labels"]]
    d.to_parquet(f"{OUT}/goemo_{name}.parquet"); print("goemo", name, len(d))
R = "brighter-dataset/BRIGHTER-emotion-categories"
langs = sorted({s.rfilename.split("/")[0] for s in HfApi().dataset_info(R).siblings if "/" in s.rfilename})
for l in langs:
    for sp in ["test", "dev"]:
        try:
            d = pd.read_parquet(hf_hub_download(R, f"{l}/{sp}-00000-of-00001.parquet", repo_type="dataset"))
        except Exception as e:
            print(l, sp, "missing", type(e).__name__); continue
        cols = [c for c in E6 if c in d.columns]
        d = d[["text"] + cols].copy()
        for c in E6:
            if c not in d:
                d[c] = -1   # emotion not annotated for this language
        if sp == "test" and len(d) > 1500:
            d = d.sample(1500, random_state=0)
        d.to_parquet(f"{OUT}/brighter/{l}_{sp}.parquet"); print(l, sp, len(d), "missing:", [c for c in E6 if c not in cols])
