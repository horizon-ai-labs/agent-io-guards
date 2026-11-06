"""Train/val/test for the NCII guard from judged pool. Label = judge P(ncii) >= 0.5 (not the generator's intent). Drop any row the judge
flags as implying a minor (j_minor >= 0.5). Confidence filter: keep rows where the judge is decided (p<=.2 or p>=.8) OR a real edit.
Val = 3% of train groups. python ncii/build_data.py JUDGED.parquet OUT_DIR"""
import sys, os, hashlib
import pandas as pd
d = pd.read_parquet(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
n0 = len(d)
d = d[d.j_minor < 0.5].copy()
d["label"] = (d.j_ncii >= 0.5).astype(int)
dec = (d.j_ncii <= 0.2) | (d.j_ncii >= 0.8) | (d.kind == "real_edit")
d = d[dec]
te = d[d.split == "test"]; tr = d[d.split == "train"].copy()
vh = tr.group.map(lambda g: int(hashlib.md5(str(g).encode()).hexdigest(), 16) % 33 == 0)
va = tr[vh]; tr = tr[~vh]
for name, part in [("train", tr), ("val", va), ("test", te)]:
    part.assign(source=part.kind)[["text", "label", "lang", "kind", "src", "source", "j_ncii"]].to_parquet(f"{out}/{name}.parquet", index=False)
print(f"dropped minor+undecided: {n0}->{len(d)}; train {len(tr)} (pos {tr.label.mean():.3f}) val {len(va)} test {len(te)} (pos {te.label.mean():.3f})")
print("test by kind:", te.groupby("kind").label.agg(["size","mean"]).round(3).to_dict("index"))
print("langs", tr.lang.nunique(), "| test langs", te.lang.nunique())
