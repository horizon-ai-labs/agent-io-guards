"""Sentiment training data v0 from teacher-labelled texts.   python sentiment/build_sent.py OUT LABELLED.parquet [...]
Inputs: sentiment/teacher_sent.py LABEL_OUT files (text, lang, src, p_neg, p_neu, p_pos) for the FineWeb-2 pool and the
synthetic texts. Removes any text that also occurs in sentiment/evals (normalised exact match) and near-empty texts; caps
confidently-neutral web snippets (teacher P(neutral) > NEU_MAX_P) at NEU_SHARE of the web share, so the data is not
dominated by informational text. OUT/train.parquet, val.parquet (3%), stats.json.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd

OUT, ins = sys.argv[1], sys.argv[2:]
os.makedirs(OUT, exist_ok=True)
NEU_MAX_P, NEU_SHARE = float(os.environ.get("NEU_MAX_P", 0.9)), float(os.environ.get("NEU_SHARE", 0.5))
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet"))
      for t in pd.read_parquet(p).text}
df = pd.concat([pd.read_parquet(p) for p in ins], ignore_index=True)
st = dict(n_in=len(df))
df = df[df.text.str.len() >= 3].drop_duplicates("text")
k = df.text.map(norm)
st["removed_eval_overlap"] = int(k.isin(ev).sum()); df = df[~k.isin(ev)]
web = df.src.str.startswith("fw")
neu = web & (df.p_neu > NEU_MAX_P)
cap = int(NEU_SHARE * web.sum())
if neu.sum() > cap:
    drop = df[neu].sample(int(neu.sum()) - cap, random_state=0).index
    st["dropped_confident_neutral_web"] = len(drop); df = df.drop(drop)
df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
nv = int(0.03 * len(df))
df.iloc[nv:].to_parquet(f"{OUT}/train.parquet"); df.iloc[:nv].to_parquet(f"{OUT}/val.parquet")
arg = np.array(["negative", "neutral", "positive"])[df[["p_neg", "p_neu", "p_pos"]].values.argmax(1)]
st.update(n_train=len(df) - nv, n_val=nv, by_src=df.src.value_counts().to_dict(),
          label_by_src=pd.crosstab(df.src, arg).to_dict("index"), n_lang=int(df.lang.nunique()))
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1, default=int); print(json.dumps(st, default=int)[:2000])
