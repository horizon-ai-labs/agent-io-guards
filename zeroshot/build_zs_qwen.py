"""Zero-shot data v1.4 candidate: v1.3 pairs with teacher soft labels from the large model (948e0c42/zs_soft) where a third of
the rows (every 3rd) additionally have a Qwen3.8-27B P(Yes) score (zeroshot/judge_zs.py label mode).
python zeroshot/build_zs_qwen.py ZS_SOFT_DIR QWEN_1,QWEN_2,... OUT     env W (default 1.0): soft = W*qwen + (1-W)*large on those rows
Why: Qwen as a zero-shot judge beats our large model on SIB/MASSIVE/B77/MTOP/CLINC samples (MTOP .64 vs .42).
"""
import os, shutil, sys
import pandas as pd

SRC, Q, OUT = sys.argv[1:4]
W = float(os.environ.get("W", 1.0))
os.makedirs(OUT, exist_ok=True)
tr = pd.read_parquet(f"{SRC}/train.parquet")
q = pd.concat([pd.read_parquet(f) for f in Q.split(",")])
print("train", len(tr), "qwen-labelled", len(q), "W", W)
s = tr.soft.values.copy()
s[q.row.values] = W * q.qwen.values + (1 - W) * s[q.row.values]
m = tr.iloc[q.row.values]
print("agreement at 0.5 on labelled rows: qwen vs hard", round(((q.qwen.values > 0.5) == (m.label.values == 1)).mean(), 3),
      "| qwen vs large", round(((q.qwen.values > 0.5) == (m.soft.values > 0.5)).mean(), 3))
tr["soft"] = s
tr.to_parquet(f"{OUT}/train.parquet")
shutil.copy(f"{SRC}/val.parquet", f"{OUT}/val.parquet")
