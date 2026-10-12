"""Merge generated irony texts (irony/gen_irony.py shards), dedupe (text, context), drop eval overlap, and add a no-context copy of 30% of
thread replies (the teacher labels it separately, so the model also learns replies without their post).
python irony/build_gen.py OUT.parquet SHARD.jsonl..."""
import glob, json, os, re, sys
import numpy as np, pandas as pd
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text} - {""}
rows = []
for p in sys.argv[2:]:
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
d = pd.DataFrame(rows); n0 = len(d); k = d.text.map(norm)
d = d[(k.str.len() >= 2) & ~k.isin(ev) & ~((d.context != "") & d.context.map(norm).isin(ev))].drop_duplicates(["text", "context"])
th = d[d.context != ""]; cp = th.sample(frac=0.3, random_state=0).assign(context="")
d = pd.concat([d, cp], ignore_index=True).drop_duplicates(["text", "context"]).reset_index(drop=True); d["source"] = "gen"
print("rows", n0, "->", len(d), "(no-context copies", len(cp), ") with context", int((d.context != "").sum()))
print(d.intended.value_counts().head(4).to_dict(), d.lang.nunique(), "langs")
d.to_parquet(sys.argv[1])
