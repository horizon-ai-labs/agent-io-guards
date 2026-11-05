"""Quality filter + train/val split of aligned teacher labels.
Keep: teacher stopped normally, faithful >= .9 (the teacher only deleted words), >= 30 units, 0.10 <= kept <= 0.95.
Val = 2% by id hash. python compress/filter_split.py ALIGNED.parquet OUT_DIR"""
import sys, os
import pandas as pd
d = pd.read_parquet(sys.argv[1]); out = sys.argv[2]; os.makedirs(out, exist_ok=True)
n0 = len(d)
d = d[(d.finish == "stop") & (d.faithful >= .9) & (d.n_units >= 30) & d.kept.between(.10, .95)]
va = d[d.id % 50 == 0]; tr = d[d.id % 50 != 0]
tr.to_parquet(f"{out}/train.parquet", index=False); va.to_parquet(f"{out}/val.parquet", index=False)
per = d.groupby("lang").size()
print(f"kept {len(d)} of {n0} ({len(d) / n0:.1%}); train {len(tr)} val {len(va)}; langs {d.lang.nunique()}; kept-ratio mean {d.kept.mean():.3f}")
print("fewest:", per.sort_values().head(10).to_dict())
