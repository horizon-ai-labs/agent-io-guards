"""Real social-media texts for hate v2 (teacher-labelled later; the datasets' own labels are NOT used):
- Measuring Hate Speech (CC-BY-4.0) comments that are not in the 6,000-comment eval sample (hate/evals/mhs.parquet);
- HateXplain (CC-BY-4.0) train + validation posts (the test split stays the eval set).
Eval overlap (all hate/evals sets) removed by normalised exact match.   python hate/build_pool2.py OUT.parquet"""
import glob, json, os, re, sys, urllib.request
import pandas as pd
from huggingface_hub import hf_hub_download
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
m = pd.read_parquet(hf_hub_download("ucberkeley-dlab/measuring-hate-speech", "measuring-hate-speech.parquet", repo_type="dataset"), columns=["comment_id", "text"])
m = m.drop_duplicates("comment_id")[["text"]].assign(source="mhs", lang="English")
B = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
ds = json.load(urllib.request.urlopen(B + "dataset.json")); sp = json.load(urllib.request.urlopen(B + "post_id_divisions.json"))
h = pd.DataFrame([dict(text=" ".join(ds[p]["post_tokens"]), source="hatexplain", lang="English") for p in sp["train"] + sp["val"]])
d = pd.concat([m, h], ignore_index=True); k = d.text.map(norm)
d = d[~k.isin(ev) & (k.str.len() >= 3)].drop_duplicates("text").reset_index(drop=True)
d.to_parquet(sys.argv[1]); print(len(d), d.source.value_counts().to_dict(), "eval overlap removed", int(k.isin(ev).sum()))
