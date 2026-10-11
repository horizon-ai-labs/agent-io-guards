"""Real-comment pool for hate-speech training: Civil Comments (CC0) and their machine translations from the toxicity v1.0 data
(tox/build_tox.py): 60k rows with identity_attack >= 0.2 + 60k other random rows; eval overlap removed. Labels come from the
teacher (hate/teacher_hate.py label).   python hate/build_pool.py TOX_V0_TRAIN.parquet OUT.parquet"""
import glob, os, re, sys
import pandas as pd
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
v = pd.read_parquet(sys.argv[1]); v = v[~v.text.map(norm).isin(ev)]
ia = v.identity_attack >= 0.2
d = pd.concat([v[ia].sample(60000, random_state=0), v[~ia].sample(60000, random_state=0)])[["text", "lang"]].assign(source="civil")
d = d.sample(frac=1, random_state=0).reset_index(drop=True); d.to_parquet(sys.argv[2]); print(len(d), d.lang.value_counts().head(5).to_dict())
