"""Scam-detector training data from teacher-labelled texts.   python scam/build_scam.py OUT LABELLED.parquet [...]
Removes texts that occur in scam/evals (normalised exact match) and duplicates; OUT/train.parquet, val.parquet (3%), stats.json."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
OUT, ins = sys.argv[1], sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
df = pd.concat([pd.read_parquet(p) for p in ins], ignore_index=True)
st = dict(n_in=len(df)); df = df[df.text.str.len() >= 5].drop_duplicates("text")
k = df.text.map(norm); st["removed_eval_overlap"] = int(k.isin(ev).sum()); df = df[~k.isin(ev)]
df = df.sample(frac=1.0, random_state=0).reset_index(drop=True); nv = int(0.03 * len(df))
df.iloc[nv:].to_parquet(f"{OUT}/train.parquet"); df.iloc[:nv].to_parquet(f"{OUT}/val.parquet")
arg = np.array(["legitimate", "spam", "fraud"])[df[["p_legit", "p_spam", "p_fraud"]].values.argmax(1)]
st.update(n_train=len(df) - nv, n_val=nv, label_by_src=pd.crosstab(df.src, arg).to_dict("index"), n_lang=int(df.lang.nunique()))
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1, default=int); print(json.dumps(st, default=int))
