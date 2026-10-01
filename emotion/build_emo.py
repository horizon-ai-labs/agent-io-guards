"""Emotion training data: go_emotions train (English, human labels) + Qwen translations (labels carried over).
python emotion/build_emo.py OUT goemo_train.parquet TR_DIR [TR_DIR ...]
Split by source comment: 3% of go_emotions train comments go to val in every language (no translation of a val comment
is in train). Texts that also occur in emotion/evals are removed. OUT/train.parquet, val.parquet (text, labels, lang, src_idx).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd

OUT, src_f, trs = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
E = os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals")
ev = {norm(t) for p in glob.glob(f"{E}/*.parquet") + glob.glob(f"{E}/brighter/*.parquet") for t in pd.read_parquet(p).text}
src = pd.read_parquet(src_f).assign(lang="English", src_idx=lambda d: range(len(d)))
fs = [f for d in trs for f in sorted(glob.glob(f"{d}/*.jsonl"))]
tr = pd.concat([pd.read_json(f, lines=True) for f in fs], ignore_index=True) if fs else pd.DataFrame(columns=["text", "labels", "lang", "src_idx"])
df = pd.concat([src, tr[["text", "labels", "lang", "src_idx"]]], ignore_index=True)
st = dict(n_src=len(src), n_translated=len(tr), per_lang=tr.lang.value_counts().to_dict())
k = df.text.map(norm); st["removed_eval_overlap"] = int(k.isin(ev).sum()); df = df[~k.isin(ev)].drop_duplicates("text")
rng = np.random.RandomState(0); val_idx = set(rng.choice(len(src), int(0.03 * len(src)), replace=False).tolist())
isv = df.src_idx.isin(val_idx)
df[~isv].sample(frac=1.0, random_state=0).to_parquet(f"{OUT}/train.parquet"); df[isv].to_parquet(f"{OUT}/val.parquet")
st.update(n_train=int((~isv).sum()), n_val=int(isv.sum()), n_lang=int(df.lang.nunique()))
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1); print(json.dumps(st)[:1500])
