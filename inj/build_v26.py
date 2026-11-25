"""Injection data v2.6 = v2.5 (data of small v2.3) + terse machine/ack negatives and their contrast positives (inj/gen_terse.py, TRAIN
family only). Hard targets (soft = label): the negatives are benign by construction; the positives are training-set attacks wrapped in
status lines / envelopes. Negatives get 2 copies (~3% of rows) to outweigh the short-command positives of v2.2/v2.3; 3% of the new rows
go to val. The TEST family (test_neg / test_pos) is never used here.
python inj/build_v26.py V25_DIR TERSE_DIR OUT"""
import os, sys
import pandas as pd
v25, terse, out = sys.argv[1:4]; os.makedirs(out, exist_ok=True)
new = pd.concat([pd.read_parquet(f"{terse}/train_neg.parquet"), pd.read_parquet(f"{terse}/train_pos.parquet")], ignore_index=True)
new = new.assign(soft=new.label.astype(float), qwen=float("nan"), soft_large=float("nan")).drop(columns=["env"])
new = new.sample(frac=1.0, random_state=0).reset_index(drop=True); k = int(len(new) * 0.03)
nv, nt = new.iloc[:k], new.iloc[k:]
nt = pd.concat([nt, nt[nt.label == 0]], ignore_index=True)
tr = pd.concat([pd.read_parquet(f"{v25}/train.parquet"), nt], ignore_index=True).sample(frac=1.0, random_state=0)
va = pd.concat([pd.read_parquet(f"{v25}/val.parquet"), nv], ignore_index=True)
tr.to_parquet(f"{out}/train.parquet"); va.to_parquet(f"{out}/val.parquet")
print("train", len(tr), "(+%d terse)" % len(nt), "val", len(va), "(+%d)" % len(nv))
