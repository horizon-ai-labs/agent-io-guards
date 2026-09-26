"""Content-safety data v3 = v2 + Civil Comments toxicity (google/civil_comments, CC0-1.0).

python safety/build_safety_v3.py V2_DIR EVAL_DIR OUT [N]
Why: teacher relabelling (v2) diluted classic toxicity: textdetox F1 .75 (v0) -> .62, recall .52 (the teacher itself is
weak there). Civil Comments has annotator fractions: soft target = toxicity for N comments with toxicity >= 0.5 and N with
toxicity == 0 (train shard 0). Categories from subtype fractions >= 0.5: obscene -> profanity, threat -> violence,
insult -> harassment, identity_attack -> hate, sexual_explicit -> sexual. Rows matching a benchmark text are removed.
"""
import glob, os, re, shutil, sys
import pandas as pd
from huggingface_hub import hf_hub_download

V2, EV, OUT = sys.argv[1:4]
N = int(sys.argv[4]) if len(sys.argv) > 4 else 60000
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
bench = set()
for p in glob.glob(f"{EV}/*.parquet"):
    bench |= {norm(x) for x in pd.read_parquet(p).text}
cc = pd.read_parquet(hf_hub_download("google/civil_comments", "data/train-00000-of-00002.parquet", repo_type="dataset"))
cc = cc[cc.text.str.len().between(10, 3000) & ~cc.text.map(norm).isin(bench)]
tox = cc[cc.toxicity >= 0.5].sample(n=min(N, int((cc.toxicity >= 0.5).sum())), random_state=0)
clean = cc[cc.toxicity == 0].sample(n=N, random_state=0)
SUB = {"obscene": "profanity", "threat": "violence", "insult": "harassment", "identity_attack": "hate", "sexual_explicit": "sexual"}
rows = pd.concat([tox, clean])
cats = [[c for k, c in SUB.items() if getattr(r, k) >= 0.5] for r in rows.itertuples()]
add = pd.DataFrame(dict(text=rows.text.values, text_pair="", unsafe=(rows.toxicity >= 0.5).astype(int).values, soft=rows.toxicity.values,
                        cats=cats, has_cats=True, lang="en", kind="prompt", tag="civil_comments", id=""))
tr = pd.read_parquet(f"{V2}/train.parquet")
out = pd.concat([tr, add], ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{OUT}/train.parquet")
for f in ["val.parquet", "test.parquet", "stats.json"]:
    shutil.copy(f"{V2}/{f}", f"{OUT}/{f}")
print("civil comments", len(add), "toxic", len(tox), "| category counts", pd.Series([c for x in cats for c in x]).value_counts().to_dict())
print("train", len(out))
