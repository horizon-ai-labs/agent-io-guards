"""Injection data v2.8b = v2.8 train + one extra copy of the two negative classes that failed the v2.8 gates (inj/gen_tool3.py train
neg_mention, inj/gen_tool2.py train neg_json; only rows already in v2.8 train, i.e. not the 3% val share). Val = v2.8 val.
python inj/build_v28b.py V28_DIR OUT"""
import os, sys
import pandas as pd
v28, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
tr = pd.read_parquet(f"{v28}/train.parquet")
extra = tr[tr.source.isin(["tool3/train_neg_mention", "tool2/train_neg_json"])]
pd.concat([tr, extra], ignore_index=True).sample(frac=1.0, random_state=0).to_parquet(f"{out}/train.parquet")
pd.read_parquet(f"{v28}/val.parquet").to_parquet(f"{out}/val.parquet")
print("train", len(tr) + len(extra), "extra", len(extra), extra.source.value_counts().to_dict())
