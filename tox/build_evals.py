"""Toxicity evaluation sets (evaluation only).   python tox/build_evals.py OUT_DIR
- civil_test.parquet: google/civil_comments test (CC0), 20,000 random comments; 7 fractional labels (toxicity, severe_toxicity,
  obscene, threat, insult, identity_attack, sexual_explicit), binarised at >= 0.5 for AUC (the Jigsaw/Detoxify convention).
- tdx_<lang>.parquet: textdetox/multilingual_toxicity_dataset (15 languages, binary toxic; OpenRAIL++, evaluation only),
  up to 1,000 per language, class-balanced where possible.
"""
import os, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
L7 = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
d = pd.read_parquet(hf_hub_download("google/civil_comments", "data/test-00000-of-00001.parquet", repo_type="dataset"))
d = d.sample(20000, random_state=0)[["text"] + L7]; d.to_parquet(f"{OUT}/civil_test.parquet")
print("civil", len(d), {l: int((d[l] >= 0.5).sum()) for l in L7})
R = "textdetox/multilingual_toxicity_dataset"
for f in [s.rfilename for s in HfApi().dataset_info(R).siblings if s.rfilename.endswith(".parquet")]:
    lang = f.split("/")[-1].split("-")[0]
    t = pd.read_parquet(hf_hub_download(R, f, repo_type="dataset")).dropna(subset=["text"]).drop_duplicates("text")
    t = pd.concat([g.sample(min(len(g), 500), random_state=0) for _, g in t.groupby("toxic")])
    t[["text", "toxic"]].to_parquet(f"{OUT}/tdx_{lang}.parquet"); print(lang, len(t), int(t.toxic.sum()))
