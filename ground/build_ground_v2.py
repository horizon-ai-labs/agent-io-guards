"""Groundedness data v2 = v1 + genre-targeted claim data (gen_ground_v2.py) + WANLI (CC-BY-4.0).
WANLI: entailment -> supported (1); neutral / contradiction -> unsupported (0).
5% of round-2 documents -> eval/synthetic_claims.  env: V1=<ground_v1 dir> GEN2=<glob ground2_*.jsonl> OUT=<dir>"""
import glob, json, os, random
import pandas as pd
from datasets import load_dataset
V1, OUT = os.environ["V1"], os.environ["OUT"]; os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(5)
tr, ev, docs = [], [], 0
for f in sorted(glob.glob(os.environ["GEN2"])):
    for line in open(f):
        r = json.loads(line); docs += 1
        rows = [dict(text=r["doc"], text_pair=c["text"], label=int(c["label"]), source=f"claims2/{r['genre']}", lang=r["lang"]) for c in r["claims"]]
        (ev if rng.random() < 0.05 else tr).extend(rows)
w = 0
for r in load_dataset("alisawuffles/WANLI", split="train"):
    tr.append(dict(text=r["premise"], text_pair=r["hypothesis"], label=int(r["gold"] == "entailment"), source="wanli", lang="English")); w += 1
t1 = pd.read_parquet(f"{V1}/train.parquet"); v1 = pd.read_parquet(f"{V1}/val.parquet")
new = pd.DataFrame(tr).drop_duplicates(["text", "text_pair"]).sample(frac=1.0, random_state=6)
nval = 2000; cols = ["text", "text_pair", "label", "source", "lang"]
pd.concat([t1[cols], new.iloc[nval:][cols]]).sample(frac=1.0, random_state=7).to_parquet(f"{OUT}/train.parquet")
pd.concat([v1[cols], new.iloc[:nval][cols]]).to_parquet(f"{OUT}/val.parquet")
for p in glob.glob(f"{V1}/eval/*.parquet"):
    pd.read_parquet(p).to_parquet(f"{OUT}/eval/{os.path.basename(p)}")
pd.DataFrame(ev).to_parquet(f"{OUT}/eval/synthetic_claims.parquet")
print(dict(docs=docs, new=len(new), wanli=w, eval=len(ev), train_total=len(t1) + len(new) - nval, pos=float(new.label.mean())), flush=True)
