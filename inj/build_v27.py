"""Injection data v2.7 = v2.6 + inj/gen_tool2.py TRAIN family (id-like negatives x2, benign-imperative JSON negatives, agent-directed
instructions in logs and in JSON fields as positives; hard targets). 3% of the new rows go to val, so seed selection (best val F1) now
sees tool-output rows of every kind (lesson from v2.6, where the selected seed was the worst on benign tool descriptions).
The TEST family is never used here.   python inj/build_v27.py V26_DIR TOOL2_DIR OUT"""
import glob, os, sys
import pandas as pd
v26, t2, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
new = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{t2}/train_*.parquet"))], ignore_index=True)
new = new.assign(soft=new.label.astype(float), qwen=float("nan"), soft_large=float("nan"))
new = new.sample(frac=1.0, random_state=0).reset_index(drop=True); k = int(len(new) * 0.03)
nv, nt = new.iloc[:k], new.iloc[k:]
nt = pd.concat([nt, nt[nt.kind == "neg_id"]], ignore_index=True)
tr = pd.concat([pd.read_parquet(f"{v26}/train.parquet"), nt], ignore_index=True).sample(frac=1.0, random_state=0)
va = pd.concat([pd.read_parquet(f"{v26}/val.parquet"), nv], ignore_index=True)
tr.to_parquet(f"{out}/train.parquet"); va.to_parquet(f"{out}/val.parquet")
print("train", len(tr), "(+%d tool2)" % len(nt), "val", len(va), "(+%d)" % len(nv), nt.kind.value_counts().to_dict())
