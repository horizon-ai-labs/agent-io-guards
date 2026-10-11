"""Hate-speech evaluation sets (evaluation only, never trained on).   python hate/build_evals.py OUT_DIR
Every file: text, hate (0/1) [+ functionality / target columns].
- mhc_<lang>.parquet: Multilingual HateCheck + HateCheck (mteb/multi-hatecheck, CC-BY-4.0): functional tests in 11 languages
  (ara cmn deu eng fra hin ita nld pol por spa), all items.
- mhs.parquet: Measuring Hate Speech (ucberkeley-dlab/measuring-hate-speech, CC-BY-4.0), English social-media comments; one row per
  comment, hate = IRT hate_speech_score > 0.5 (the authors' "hateful" cut); target_<group> = share of annotators naming the group
  >= 0.5. 6,000 random comments.
- hatexplain.parquet: HateXplain test split (github.com/hate-alert/HateXplain, CC-BY-4.0), English tweets / Gab posts; hate = majority
  "hatespeech" (offensive and normal = 0); offensive = majority "offensive".
- kmhas.parquet: K-MHaS test, 5,000 random (github.com/adlnlp/K-MHaS, as in jeanlee/kmhas_korean_hate_speech, CC-BY-SA-4.0), Korean news comments; hate = any protected category
  (origin, physical, age, gender, race, religion); 0 = not_hate_speech; items labelled only politics/profanity dropped.
"""
import json, os, sys, urllib.request
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
d = pd.concat([pd.read_json(hf_hub_download("mteb/multi-hatecheck", f"test/{l}.jsonl.gz", repo_type="dataset"), lines=True)
               for l in "ara cmn deu eng fra hin ita nld pol por spa".split()])
d["hate"] = (d.is_hateful == "hateful").astype(int)
for l, g in d.groupby("lang"):
    g[["text", "hate", "functionality"]].reset_index(drop=True).to_parquet(f"{OUT}/mhc_{l}.parquet"); print("mhc", l, len(g), int(g.hate.sum()))

m = pd.read_parquet(hf_hub_download("ucberkeley-dlab/measuring-hate-speech", "measuring-hate-speech.parquet", repo_type="dataset"))
G = ["race", "religion", "origin", "gender", "sexuality", "age", "disability"]
agg = m.groupby("comment_id").agg(text=("text", "first"), score=("hate_speech_score", "first"), **{f"target_{g}": (f"target_{g}", "mean") for g in G})
agg = agg.sample(6000, random_state=0); agg["hate"] = (agg.score > 0.5).astype(int)
for g in G:
    agg[f"target_{g}"] = (agg[f"target_{g}"] >= 0.5).astype(int)
agg.reset_index(drop=True).to_parquet(f"{OUT}/mhs.parquet"); print("mhs", len(agg), int(agg.hate.sum()), {g: int(agg[agg.hate == 1][f"target_{g}"].sum()) for g in G})

B = "https://raw.githubusercontent.com/hate-alert/HateXplain/master/Data/"
ds = json.load(urllib.request.urlopen(B + "dataset.json")); sp = json.load(urllib.request.urlopen(B + "post_id_divisions.json"))
rows = []
for pid in sp["test"]:
    labs = [a["label"] for a in ds[pid]["annotators"]]; maj = max(set(labs), key=labs.count)
    if labs.count(maj) >= 2:
        rows.append(dict(text=" ".join(ds[pid]["post_tokens"]), hate=int(maj == "hatespeech"), offensive=int(maj == "offensive")))
h = pd.DataFrame(rows); h.to_parquet(f"{OUT}/hatexplain.parquet"); print("hatexplain", len(h), int(h.hate.sum()), int(h.offensive.sum()))

lines = urllib.request.urlopen("https://raw.githubusercontent.com/adlnlp/K-MHaS/main/data/kmhas_test.txt").read().decode().splitlines()[1:]
k = pd.DataFrame([dict(text=r[0], label=[int(x) for x in r[1].split(",")]) for r in (l.strip().split("\t") for l in lines) if len(r) == 2])
P = {0, 1, 4, 5, 6, 7}
k["hate"] = k.label.map(lambda L: int(bool(set(L) & P))); k = k[(k.hate == 1) | k.label.map(lambda L: list(L) == [8])]
k = k.drop_duplicates("text").sample(5000, random_state=0); k[["text", "hate"]].reset_index(drop=True).to_parquet(f"{OUT}/kmhas.parquet"); print("kmhas", len(k), int(k.hate.sum()))
