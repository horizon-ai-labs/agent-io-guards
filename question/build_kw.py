"""Keyword-query vs natural-language data (drop-in labels of shahrukhx01/bert-mini-finetune-question-detection: LABEL_0 = keyword query,
LABEL_1 = natural-language question or statement), from the question/build_data.py splits (objective labels, no LLM):
keywords = the reranker's keyword queries; natural language = its question queries + web sentences (questions and statements).
python question/build_kw.py Q_DATA_DIR Q_EVAL_DIR OUT_DIR EVAL_DIR"""
import json, os, sys
import numpy as np, pandas as pd
qd, qe, out, ev = sys.argv[1:5]
def conv(df, rng, n_sent):
    q = df[df.source == "query"]; s = df[df.source == "sentence"]
    s = s.sample(min(len(s), n_sent), random_state=rng)
    return pd.concat([q.assign(label=q.label.values), s.assign(label=1)], ignore_index=True)  # query label: 1 = question, 0 = keywords
os.makedirs(out, exist_ok=True); os.makedirs(ev, exist_ok=True)
tr = pd.read_parquet(f"{qd}/train.parquet"); va = pd.read_parquet(f"{qd}/val.parquet")
ntr = int((tr[tr.source == "query"].label == 0).sum())  # as many sentences as keyword queries
conv(tr, 0, ntr).sample(frac=1, random_state=0).to_parquet(f"{out}/train.parquet"); conv(va, 0, int((va[va.source == "query"].label == 0).sum())).to_parquet(f"{out}/val.parquet")
qt = pd.read_parquet(f"{qe}/queries.parquet"); st = pd.read_parquet(f"{qe}/sentences.parquet")
qt.to_parquet(f"{ev}/queries.parquet")                                                     # keywords (0) vs questions (1)
pd.concat([qt[qt.label == 0], st.assign(label=1).groupby("lang").head(60)], ignore_index=True).to_parquet(f"{ev}/kw_vs_sentences.parquet")  # keywords (0) vs web sentences (1)
t = pd.read_parquet(f"{out}/train.parquet"); print(json.dumps(dict(train=len(t), pos=round(float(t.label.mean()), 3), by_source=t.source.value_counts().to_dict())))
