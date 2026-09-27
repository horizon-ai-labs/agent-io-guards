"""Groundedness data v4 = v3 pairs + a soft label from the Qwen3.8-27B judge (ground/judge_qwen.py, P(Yes)).
python ground/build_ground_v4.py V3_DIR JUDGED_1,JUDGED_2,... OUT
train.parquet gets a `soft` column (judge P(supported)); train/train.py --distill A then uses (1-A)·CE(hard) + A·CE(soft).
Why: the judge scores AggreFact .752 mean bacc (MiniCheck-R .749, our base v1.1 .716; AUC mostly .83-.96).
"""
import os, shutil, sys
import pandas as pd

V3, J, OUT = sys.argv[1:4]
os.makedirs(OUT, exist_ok=True)
tr = pd.read_parquet(f"{V3}/train.parquet")
j = pd.concat([pd.read_parquet(f, columns=["row", "judge"]) for f in J.split(",")]).sort_values("row")
assert len(j) == len(tr) and (j.row.values == range(len(tr))).all(), "missing judge shards"
tr["soft"] = j.judge.values
agree = ((tr.soft > 0.5) == (tr.label == 1))
print("judge vs construction labels: agree", round(agree.mean(), 3))
print(tr.assign(a=agree).groupby("source").a.mean().sort_values().head(15).round(3).to_string())
tr.to_parquet(f"{OUT}/train.parquet")
shutil.copy(f"{V3}/val.parquet", f"{OUT}/val.parquet")
if os.path.isdir(f"{V3}/eval"):
    shutil.copytree(f"{V3}/eval", f"{OUT}/eval", dirs_exist_ok=True)
