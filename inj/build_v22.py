"""Injection guard v2.2 training data: v2 train/val with teacher soft = max(injection-large soft score, Qwen3.8-27B judge).
python inj/build_v22.py OUT QWEN_SHARD.parquet [...]
Why max: on the eval suite, max(large, qwen) at 0.5 matched or beat the large model on almost every set (Qwen adds true
positives with very few false positives); see COMPANY_STATE (injection v2.2 plan).
"""
import json, os, sys
import numpy as np, pandas as pd
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
df = pd.concat([pd.read_parquet(p) for p in sys.argv[2:]], ignore_index=True)
df["soft_large"] = df.soft; df["soft"] = np.maximum(df.soft.values, df.qwen.values)
st = dict(n=len(df), raised=int((df.soft > df.soft_large + 0.2).sum()), agree_hard=float(((df.soft > 0.5) == (df.label == 1)).mean()),
          agree_hard_large=float(((df.soft_large > 0.5) == (df.label == 1)).mean()),
          qwen_pos_on_label0=float((df.qwen[df.label == 0] > 0.5).mean()), qwen_pos_on_label1=float((df.qwen[df.label == 1] > 0.5).mean()))
for sp in ["train", "val"]:
    df[df.split == sp].drop(columns=["split"]).to_parquet(f"{OUT}/{sp}.parquet")
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1); print(st)
