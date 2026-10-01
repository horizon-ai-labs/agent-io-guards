"""Toxicity training data: Civil Comments English sample + Qwen translations (soft labels carried over).
python tox/build_tox.py OUT civil_train_sample.parquet TR_DIR [N_EN]
English: N_EN comments (half with toxicity >= 0.2). Val = 2% of source comments (all their translations too).
Texts also in tox/evals are removed. OUT/train.parquet, val.parquet (text, lang, src_idx, 7 label columns).
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
OUT, src_f, trd = sys.argv[1:4]; NEN = int(sys.argv[4]) if len(sys.argv) > 4 else 300000
os.makedirs(OUT, exist_ok=True)
L7 = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
src = pd.read_parquet(src_f); src["src_idx"] = src.index
tr = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(f"{trd}/*.jsonl"))], ignore_index=True)
used = set(tr.src_idx)
tox = src[src.toxicity >= 0.2]; non = src[src.toxicity < 0.2]
en = pd.concat([src[src.src_idx.isin(used)], tox.sample(NEN // 2, random_state=1), non.sample(NEN // 2, random_state=1)]).drop_duplicates("src_idx").assign(lang="English")
df = pd.concat([en[["text", "lang", "src_idx"] + L7], tr[["text", "lang", "src_idx"] + L7]], ignore_index=True)
st = dict(n_en=len(en), n_translated=len(tr), per_lang=tr.lang.value_counts().to_dict())
k = df.text.map(norm); st["removed_eval_overlap"] = int(k.isin(ev).sum()); df = df[~k.isin(ev)].drop_duplicates("text")
rng = np.random.RandomState(0); vs = set(rng.choice(df.src_idx.unique(), int(0.02 * df.src_idx.nunique()), replace=False).tolist())
isv = df.src_idx.isin(vs)
df[~isv].sample(frac=1.0, random_state=0).to_parquet(f"{OUT}/train.parquet"); df[isv].to_parquet(f"{OUT}/val.parquet")
st.update(n_train=int((~isv).sum()), n_val=int(isv.sum()), toxic_share=float((df.toxicity >= 0.5).mean()))
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1); print(json.dumps(st)[:800])
