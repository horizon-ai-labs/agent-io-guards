"""Groundedness data v0: (text = context, text_pair = response, label 1 supported / 0 unsupported).

env: GEN=<glob of ground_*.jsonl>  OUT=<dir>
Sources: Qwen synthetic over FineWeb/FineWeb-2 passages (split by passage: 5% -> eval/synthetic_ground),
RAGTruth train (MIT; response-level label: unsupported if any annotated hallucination).
HaluEval and RAGTruth test are evaluation-only (see ground/evaluate_ground.py).
"""
import glob, json, os, random
import pandas as pd
from datasets import load_dataset

OUT = os.environ["OUT"]; os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(1)
tr, ev = [], []
n_items = 0
for f in sorted(glob.glob(os.environ["GEN"])):
    for line in open(f):
        r = json.loads(line); n_items += 1
        ctx = (r["task"].strip() + "\n\n" + r["passage"]).strip() if r.get("task") else r["passage"]
        rows = [dict(text=ctx, text_pair=x["text"], label=int(x["label"]), source=f"synthetic/{'supported' if x['label'] else 'unsupported'}",
                     lang=r["lang"]) for x in r["responses"]]
        (ev if rng.random() < 0.05 else tr).extend(rows)
rt = 0
for r in load_dataset("wandb/RAGTruth-processed", split="train"):
    labs = r["hallucination_labels_processed"]
    hal = bool(labs) and any(v for v in (labs.values() if isinstance(labs, dict) else [labs]))
    ctx = ((r["query"] or "") + "\n\n" + (r["context"] or "")).strip()
    tr.append(dict(text=ctx, text_pair=r["output"], label=0 if hal else 1, source=f"ragtruth/{r['task_type']}", lang="English")); rt += 1
df = pd.DataFrame(tr).drop_duplicates(["text", "text_pair"]).sample(frac=1.0, random_state=2).reset_index(drop=True)
nval = 3000
df.iloc[nval:].to_parquet(f"{OUT}/train.parquet"); df.iloc[:nval].to_parquet(f"{OUT}/val.parquet")
pd.DataFrame(ev).to_parquet(f"{OUT}/eval/synthetic_ground.parquet")
print(dict(items=n_items, train=len(df) - nval, val=nval, eval=len(ev), ragtruth=rt, pos_rate=float(df.label.mean()),
           by_source=df.source.value_counts().to_dict(), langs=df.lang.value_counts().head(10).to_dict()), flush=True)
