"""Prompt classifier training data: teacher-labelled prompts (prompt/teacher_prompt.py on prompt/build_label_in.py output); rows with a
missing label dropped; test prompts removed (normalised exact match); val = 5% random.   python prompt/build_data.py OUT_DIR SHARDS..."""
import json, os, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import TASKS, DIMS
L = [f"task_{t}" for t in TASKS] + DIMS
out = sys.argv[1]; d = pd.concat([pd.read_parquet(p) for p in sys.argv[2:]], ignore_index=True); n0 = len(d)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = set(pd.read_parquet(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "test_judged.parquet")).text.map(norm))
d = d.dropna(subset=L); k = d.text.map(norm); n_ev = int(k.isin(ev).sum()); d = d[~k.isin(ev)]
d["val"] = np.random.RandomState(0).rand(len(d)) < 0.05; os.makedirs(out, exist_ok=True); cols = ["text", "lang", "source"] + L
d[~d.val][cols].sample(frac=1, random_state=0).to_parquet(f"{out}/train.parquet"); d[d.val][cols].to_parquet(f"{out}/val.parquet")
task = np.array(TASKS)[d[[f"task_{t}" for t in TASKS]].values.argmax(1)]
st = dict(n_in=n0, train=int((~d.val).sum()), val=int(d.val.sum()), removed_test_overlap=n_ev, n_lang=int(d.lang.nunique()), by_source=d.source.value_counts().to_dict(),
          task_dist=pd.Series(task).value_counts().to_dict(), dim_means={x: round(float(d[x].mean()), 3) for x in DIMS})
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
