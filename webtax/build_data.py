"""Topic & format training data from teacher label shards (webtax/teacher_webtax.py on .gated/webtax/label_in.parquet); rows with missing
labels dropped; val = 5% random.   python webtax/build_data.py OUT_DIR SHARDS..."""
import importlib.util, json, os, sys
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("t", os.path.join(HERE, "teacher_webtax.py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
out = sys.argv[1]; d = pd.concat([pd.read_parquet(p) for p in sys.argv[2:]], ignore_index=True).dropna(subset=m.LABELS)
d["val"] = np.random.RandomState(0).rand(len(d)) < 0.05; os.makedirs(out, exist_ok=True); cols = ["text", "lang", "source"] + m.LABELS
d[~d.val][cols].sample(frac=1, random_state=0).to_parquet(f"{out}/train.parquet"); d[d.val][cols].to_parquet(f"{out}/val.parquet")
T = [l for l in m.LABELS if l.startswith("topic_")]; F = [l for l in m.LABELS if l.startswith("format_")]
st = dict(train=int((~d.val).sum()), val=int(d.val.sum()), n_lang=int(d.lang.nunique()), by_source=d.source.value_counts().to_dict(),
          topic_dist=pd.Series(np.array(T)[d[T].values.argmax(1)]).value_counts().to_dict(), format_dist=pd.Series(np.array(F)[d[F].values.argmax(1)]).value_counts().to_dict())
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
