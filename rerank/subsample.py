"""Pick queries for teacher scoring.   python rerank/subsample.py MINED.parquet PASSAGES.parquet OUT.parquet [PER_LANG] [ENG] [K]
Stratified by the source passage's language: up to PER_LANG queries per language (ENG for English), keeping the
question/keywords/english mix; candidates cut to the top K (source first, then the hardest retrieved negatives)."""
import sys
import pandas as pd
m, p, out = sys.argv[1:4]
per, eng, K = (int(sys.argv[4]) if len(sys.argv) > 4 else 3200), (int(sys.argv[5]) if len(sys.argv) > 5 else 20000), (int(sys.argv[6]) if len(sys.argv) > 6 else 16)
df = pd.read_parquet(m); lang = dict(pd.read_parquet(p)[["pid", "lang"]].values); df["plang"] = df.pid.map(lang)
df = pd.concat([g.sample(min(len(g), eng if l == "eng_Latn" else per), random_state=0) for l, g in df.groupby("plang")])
df["cands"] = df.cands.map(lambda c: list(c)[:K])
df.sample(frac=1.0, random_state=0).reset_index(drop=True).to_parquet(out)
print(len(df), "queries", len(df) * K, "pairs", df.qtype.value_counts().to_dict())
