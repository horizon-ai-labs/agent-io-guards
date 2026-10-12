"""Real comments for irony training: 60k random Civil Comments (CC0) and their machine translations from the toxicity v1.0 data
(tox/build_tox.py), no context; eval overlap removed. Labels come from the teacher.   python irony/build_pool.py TOX_V0_TRAIN.parquet OUT.parquet"""
import glob, os, re, sys
import pandas as pd
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
v = pd.read_parquet(sys.argv[1], columns=["text", "lang", "src_idx"]); v = v[~v.text.map(norm).isin(ev)]
d = v.sample(60000, random_state=1).assign(context="", source="civil").reset_index(drop=True)
d.to_parquet(sys.argv[2]); print(len(d), d.lang.value_counts().head(4).to_dict())
