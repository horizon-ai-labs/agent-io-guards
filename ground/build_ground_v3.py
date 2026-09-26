"""Groundedness data v3 = v2 + multi-sentence C2D / D2C pairs (ground/gen_c2d.py).
python ground/build_ground_v3.py V2_DIR OUT_DIR GEN_DIR [GEN_DIR ...]   (2% of the new documents go to val)"""
import glob, json, os, random, shutil, sys
import pandas as pd
v2, out, gens = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
rows = [json.loads(l) for g in gens for p in glob.glob(f"{g}/c2d_*.jsonl") if "c2d_99" not in p for l in open(p)]
new = pd.DataFrame(rows)[["text", "text_pair", "label", "source", "lang"]]
new["source"] = "multisent/" + new.source
SUB = float(os.environ.get("SUB", 1.0))   # keep this fraction of the new claims (all pairs of a kept claim stay together)
if SUB < 1:
    claims = list(new.text_pair.unique()); random.Random(1).shuffle(claims)
    new = new[new.text_pair.isin(set(claims[: int(SUB * len(claims))]))]
docs = list(new.text_pair.unique()); random.Random(0).shuffle(docs); vd = set(docs[: int(0.02 * len(docs))])   # split by claim
tr = pd.concat([pd.read_parquet(f"{v2}/train.parquet"), new[~new.text_pair.isin(vd)]]).sample(frac=1.0, random_state=0)
va = pd.concat([pd.read_parquet(f"{v2}/val.parquet"), new[new.text_pair.isin(vd)]])
tr.to_parquet(f"{out}/train.parquet"); va.to_parquet(f"{out}/val.parquet")
if os.path.exists(f"{v2}/eval"):
    shutil.copytree(f"{v2}/eval", f"{out}/eval", dirs_exist_ok=True)
print("new pairs", len(new), new.groupby("source").label.agg(["count", "mean"]).to_dict(), "| train", len(tr), "val", len(va), "pos", round(tr.label.mean(), 3))
print(new.lang.value_counts().head(8).to_dict())
