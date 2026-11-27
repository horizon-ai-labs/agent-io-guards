"""Injection data v2.8 = v2.7 + inj/gen_tool3.py TRAIN family (destination-free / internal-recipient harmful actions in prose, ops frames,
conditionals and call syntax as positives; legitimate log / doc / changelog mentions of the same tools as negatives; every record pretty
AND compact; hard targets). 3% of the new rows go to val. The TEST family is never used here.   python inj/build_v28.py V27_DIR TOOL3_DIR OUT"""
import glob, os, sys
import pandas as pd
v27, t3, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
new = pd.concat([pd.read_parquet(p) for p in sorted(glob.glob(f"{t3}/train_*.parquet"))], ignore_index=True)
new = new.assign(soft=new.label.astype(float), qwen=float("nan"), soft_large=float("nan"))
new = new.sample(frac=1.0, random_state=0).reset_index(drop=True); k = int(len(new) * 0.03)
tr = pd.concat([pd.read_parquet(f"{v27}/train.parquet"), new.iloc[k:]], ignore_index=True).sample(frac=1.0, random_state=0)
va = pd.concat([pd.read_parquet(f"{v27}/val.parquet"), new.iloc[:k]], ignore_index=True)
tr.to_parquet(f"{out}/train.parquet"); va.to_parquet(f"{out}/val.parquet")
print("train", len(tr), "(+%d tool3)" % (len(new) - k), "val", len(va), "(+%d)" % k)
