"""Irony / sarcasm evaluation sets (evaluation only).   python irony/build_evals.py OUT_DIR
Every file: text, context ('' if none), label (1 = ironic/sarcastic).
- mpico_<lang>.parquet: MultiPICo (Multilingual-Perspectivist-NLU/MultiPICo, CC-BY-4.0): replies to Reddit/Twitter posts in 9 languages
  (ar de en es fr hi it nl pt), context = the post; label = majority of the annotators (ties dropped); up to 1,500 replies per language.
- semeval18.parquet: SemEval-2018 Task 3A irony (cardiffnlp/tweet_eval 'irony' test), English tweets.
- isarcasm.parquet: iSarcasmEval 2022 task A test (tlam25/iSarcasm_taskA), English tweets, sarcastic vs not.
- ar_sarcasm.parquet: ArSarcasm test (iabufarha/ar_sarcasm, MIT), Arabic tweets.
- ko_sarcasm.parquet: Korean sarcasm test (SpellOnYou/kor_sarcasm, MIT), Korean tweets.
"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
m = pd.read_csv(hf_hub_download("Multilingual-Perspectivist-NLU/MultiPICo", "multipico_anonimized.csv", repo_type="dataset"))
g = m.groupby("reply_id").agg(text=("reply", "first"), context=("post", "first"), lang=("language", "first"), y=("label", "mean"), n=("label", "size"))
g = g[(g.y != 0.5)]; g["label"] = (g.y > 0.5).astype(int)
for l, d in g.groupby("lang"):
    d = d.sample(min(len(d), 1500), random_state=0)
    d[["text", "context", "label"]].reset_index(drop=True).to_parquet(f"{OUT}/mpico_{l}.parquet"); print("mpico", l, len(d), int(d.label.sum()))
rd = lambda r, f: pd.read_parquet(hf_hub_download(r, f, repo_type="dataset"))
from huggingface_hub import HfApi
fl = lambda r: HfApi().list_repo_files(r, repo_type="dataset")
t = rd("cardiffnlp/tweet_eval", [f for f in fl("cardiffnlp/tweet_eval") if f.startswith("irony/test")][0])
t.assign(context="")[["text", "context", "label"]].to_parquet(f"{OUT}/semeval18.parquet"); print("semeval18", len(t), int(t.label.sum()))
s = pd.read_csv(hf_hub_download("tlam25/iSarcasm_taskA", "test.csv", repo_type="dataset")).dropna(subset=["tweet"])
s = s.rename(columns={"tweet": "text", "sarcastic": "label"}).assign(context="")
s[["text", "context", "label"]].to_parquet(f"{OUT}/isarcasm.parquet"); print("isarcasm", len(s), int(s.label.sum()))
a = rd("iabufarha/ar_sarcasm", [f for f in fl("iabufarha/ar_sarcasm") if "test" in f and f.endswith(".parquet")][0])
a = a.rename(columns={"tweet": "text"}).assign(context="", label=lambda x: x.sarcasm.astype(int))
a[["text", "context", "label"]].to_parquet(f"{OUT}/ar_sarcasm.parquet"); print("ar_sarcasm", len(a), int(a.label.sum()))
k = rd("SpellOnYou/kor_sarcasm", [f for f in fl("SpellOnYou/kor_sarcasm") if "test" in f and f.endswith(".parquet")][0])
k = k.rename(columns={"tokens": "text"}).assign(context="")
k[["text", "context", "label"]].to_parquet(f"{OUT}/ko_sarcasm.parquet"); print("ko_sarcasm", len(k), int(k.label.sum()))
