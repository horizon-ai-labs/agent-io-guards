"""Financial sentiment evaluation sets (evaluation only, never trained on).   python fin/build_evals.py OUT_DIR
- fpb_allagree: Financial PhraseBank (CC-BY-NC-SA-3.0) all-annotators-agree sentences (mteb test split); fpb_50agree_hard:
  sentences with 50-99% annotator agreement (original release, 1,000 stratified). English news sentences.
  ProsusAI/finbert and most FinBERT-style models were trained on it (in-domain for them).
- tfns: zeroshot/twitter-financial-news-sentiment validation (MIT): finance tweets, bearish/bullish/neutral.
- fiqa: TheFinAI/fiqa-sentiment-classification test (MIT; FiQA 2018 task 1 headlines and microblogs; score <-0.1 negative,
  > 0.1 positive, else neutral).
- mfs_<lang>: Kenpache/multilingual-financial-sentiment (Apache-2.0; financial news sentences in en zh ja de fr es ar; how the
  labels were made is not documented); up to 900 per language, stratified.
Every file: text, label in {negative, neutral, positive}, binary = 0.
"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
L = {0: "negative", 1: "neutral", 2: "positive"}


def save(name, d):
    d = d.assign(binary=0, text=d.text.astype(str).str.slice(0, 2000))[["text", "label", "binary"]].dropna()
    d = d[d.text.str.strip().str.len() > 3].drop_duplicates("text")
    d.to_parquet(f"{OUT}/{name}.parquet"); print(name, len(d), d.label.value_counts().to_dict(), flush=True)


def strat(d, n, seed=0):
    if len(d) <= n:
        return d
    return pd.concat([g.sample(int(round(n * len(g) / len(d))), random_state=seed) for _, g in d.groupby("label")])


d = pd.read_parquet(hf_hub_download("mteb/financial_phrasebank", "sentences_allagree/test-00000-of-00001.parquet", repo_type="dataset"))
save("fpb_allagree", d.assign(label=d.label.map(L)))   # mteb labels: 0 negative, 1 neutral, 2 positive
# 50%-agreement sentences from the original release, minus the all-agree ones, 1,000 stratified
import io, zipfile
z = zipfile.ZipFile(hf_hub_download("takala/financial_phrasebank", "data/FinancialPhraseBank-v1.0.zip", repo_type="dataset"))
f = next(n for n in z.namelist() if n.endswith("Sentences_50Agree.txt"))
rows = [l.rsplit("@", 1) for l in z.read(f).decode("latin-1").splitlines() if "@" in l]
d = pd.DataFrame(rows, columns=["text", "label"]).assign(label=lambda x: x.label.str.strip())
fa = next(n for n in z.namelist() if n.endswith("Sentences_AllAgree.txt"))
allag = {l.rsplit("@", 1)[0] for l in z.read(fa).decode("latin-1").splitlines() if "@" in l}
save("fpb_50agree_hard", strat(d[~d.text.isin(allag)], 1000))   # sentences where annotators disagreed (harder)
d = pd.read_csv(hf_hub_download("zeroshot/twitter-financial-news-sentiment", "sent_valid.csv", repo_type="dataset"))
save("tfns", d.assign(label=d.label.map({0: "negative", 1: "positive", 2: "neutral"})))
d = pd.read_parquet(hf_hub_download("TheFinAI/fiqa-sentiment-classification", "data/test-00000-of-00001-0fb9f3a47c7d0fce.parquet", repo_type="dataset"))
print("fiqa cols", d.columns.tolist())
tcol = "sentence" if "sentence" in d else "text"
save("fiqa", d.rename(columns={tcol: "text"}).assign(label=d.score.map(lambda s: "negative" if s < -0.1 else "positive" if s > 0.1 else "neutral")))
d = pd.read_csv(hf_hub_download("Kenpache/multilingual-financial-sentiment", "all_languages_clean.csv", repo_type="dataset"))
print("mfs cols", d.columns.tolist(), d.iloc[0].to_dict())
tcol = next(c for c in ["text", "sentence", "Sentence", "headline"] if c in d); lcol = next(c for c in ["label", "sentiment", "Sentiment"] if c in d)
gcol = next(c for c in ["language", "lang", "Language"] if c in d)
d = d.rename(columns={tcol: "text"}).assign(label=d[lcol].astype(str).str.lower().str.strip())
for lang, g in d.groupby(gcol):
    save(f"mfs_{str(lang).lower()[:2]}", strat(g[g.label.isin(["negative", "neutral", "positive"])], 900))
