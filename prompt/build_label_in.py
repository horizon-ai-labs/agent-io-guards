"""Training subset of the prompt pool for teacher labelling: up to 3,000 prompts per language, English 20,000, Chinese 8,000.
python prompt/build_label_in.py TRAIN_PROMPTS.parquet OUT.parquet"""
import sys
import pandas as pd
d = pd.read_parquet(sys.argv[1]).sample(frac=1, random_state=0)
x = pd.concat([d[~d.lang.isin(["English", "Chinese"])].groupby("lang").head(3000), d[d.lang == "English"].head(20000), d[d.lang == "Chinese"].head(8000)])
x.reset_index(drop=True).to_parquet(sys.argv[2]); print(len(x), x.lang.nunique(), x.source.value_counts().to_dict())
