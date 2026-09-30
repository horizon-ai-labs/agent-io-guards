"""Sentiment evaluation sets (evaluation only, never trained on).   python sentiment/build_evals.py OUT_DIR
- tweets_<lang>: cardiffnlp/tweet_sentiment_multilingual test (SemEval-derived tweets, 8 languages; 3 classes)
- amazon_<lang>: amazon_reviews_multi test via mteb (CC-BY-NC-style licence -> eval only; 6 languages); stars 1-2 negative,
  3 neutral, 4-5 positive; 900 per language, 300 per class
- mteb_<lang>: mteb/multilingual-sentiment-classification test (binary positive/negative; 29 languages, several machine
  translated); up to 600 per language, class-balanced where possible
Every file: text, label in {negative, neutral, positive}, binary (1 if the set has no neutral class).
"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
L3 = {0: "negative", 1: "neutral", 2: "positive"}
cut = lambda s: s.str.slice(0, 2000)


def bal(df, n, seed=0):
    per = n // df.label.nunique()
    return pd.concat([g.sample(min(len(g), per), random_state=seed) for _, g in df.groupby("label")])


for lang in ["arabic", "english", "french", "german", "hindi", "italian", "portuguese", "spanish"]:
    d = pd.read_json(hf_hub_download("cardiffnlp/tweet_sentiment_multilingual", f"data/{lang}/test.jsonl", repo_type="dataset"), lines=True)
    d = d.assign(label=d.label.map(L3), binary=0, text=cut(d.text))
    d[["text", "label", "binary"]].to_parquet(f"{OUT}/tweets_{lang[:2]}.parquet"); print("tweets", lang, len(d), d.label.value_counts().to_dict())
for lang in ["de", "en", "es", "fr", "ja", "zh"]:
    d = pd.read_json(hf_hub_download("mteb/amazon_reviews_multi", f"{lang}/test.jsonl", repo_type="dataset"), lines=True)
    d = d.assign(label=d.label.map({0: "negative", 1: "negative", 2: "neutral", 3: "positive", 4: "positive"}), binary=0, text=cut(d.text))
    d = bal(d, 900)
    d[["text", "label", "binary"]].to_parquet(f"{OUT}/amazon_{lang}.parquet"); print("amazon", lang, len(d))
for lang in ["ara", "bam", "bul", "cmn", "cym", "deu", "dza", "ell", "eng", "eus", "fas", "fin", "heb", "hrv", "ind", "jpn", "kor",
             "mlt", "nor", "pol", "rus", "slk", "spa", "tha", "tur", "uig", "urd", "vie", "zho"]:
    d = pd.read_parquet(hf_hub_download("mteb/multilingual-sentiment-classification", f"test/{lang}.parquet", repo_type="dataset"))
    d = d.assign(label=d.label.map({0: "negative", 1: "positive"}), binary=1, text=cut(d.text))
    d = bal(d.drop_duplicates("text"), 600)
    d[["text", "label", "binary"]].to_parquet(f"{OUT}/mteb_{lang}.parquet"); print("mteb", lang, len(d), d.label.value_counts().to_dict())
