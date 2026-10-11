"""Merge the generated hate-speech texts (hate/gen_hate.py shards), dedupe, drop eval overlap -> OUT parquet (text, lang, category, platform, intended).
python hate/build_gen.py OUT.parquet SHARD.jsonl..."""
import json, sys, os, re, glob
import pandas as pd
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
rows = []
for p in sys.argv[2:]:
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
df = pd.DataFrame(rows); n0 = len(df)
k = df.text.map(norm); df = df[(k.str.len() >= 3) & ~k.isin(ev)].drop_duplicates("text").reset_index(drop=True)
print("rows", n0, "->", len(df), "eval overlap", int(k.isin(ev).sum()))
print(df.lang.value_counts().to_string()); print(df.intended.value_counts().head(5).to_string())
df.to_parquet(sys.argv[1])
