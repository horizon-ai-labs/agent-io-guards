"""Spam / phishing / scam evaluation sets (evaluation only, never trained on).   python scam/build_evals.py OUT_DIR
Every file: text, label (1 = spam, phishing or scam; 0 = legitimate), kind.
- phish_email: zefang-liu/phishing-email-dataset (LGPL-3.0; Kaggle phishing vs safe emails), 1,000 balanced.
- enron_spam: SetFit/enron_spam test (Enron ham vs spam emails), 1,000 balanced.
- sms_en: UCI SMS Spam Collection (ucirvine/sms_spam), 600 balanced.
- sms_<lang>: dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset (GPL; the UCI messages machine-translated into 21 languages),
  600 balanced per language (same message ids in every language).
- smish_bn: shariul-islam/bengali-sms-smishing-dataset test (MIT; Bengali, Banglish, code-mixed, English; smishing and promo = 1).
- smish_pt: MOZNLP/MOZ-Smishing test (Mozambican Portuguese SMS and Facebook messages), 1,000 balanced where possible.
- smish_ko: jmjmjm3/kor-smishing-message (CC-BY-NC-SA-4.0; Korean SMS, smishing vs normal), 1,000 balanced.
"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download as h

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)


def bal(d, n, seed=0):
    k = n // 2
    return pd.concat([g.sample(min(len(g), k), random_state=seed) for _, g in d.groupby("label")])


def save(name, d, kind):
    d = d.assign(text=d.text.astype(str).str.strip().str.slice(0, 3000), kind=kind)[["text", "label", "kind"]]
    d = d[d.text.str.len() > 3].drop_duplicates("text").sample(frac=1.0, random_state=0)
    d.to_parquet(f"{OUT}/{name}.parquet"); print(name, len(d), d.label.value_counts().to_dict(), flush=True)


d = pd.read_csv(h("zefang-liu/phishing-email-dataset", "Phishing_Email.csv", repo_type="dataset")).dropna()
save("phish_email", bal(d.rename(columns={"Email Text": "text"}).assign(label=(d["Email Type"] == "Phishing Email").astype(int)), 1000), "phishing")
d = pd.read_json(h("SetFit/enron_spam", "test.jsonl", repo_type="dataset"), lines=True)
save("enron_spam", bal(d.assign(label=d.label.astype(int)), 1000), "spam")
d = pd.read_csv(h("dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset", "data-augmented.csv", repo_type="dataset"))
d["label"] = (d.labels == "spam").astype(int)
ids = bal(d[["label"]].assign(i=d.index), 600).i
for c in [c for c in d.columns if c == "text" or c.startswith("text_")]:
    lang = "en" if c == "text" else c[5:]
    save(f"sms_{lang}", d.loc[ids, [c, "label"]].rename(columns={c: "text"}).dropna(), "spam")
d = pd.read_parquet(h("shariul-islam/bengali-sms-smishing-dataset", "data/test-00000-of-00001.parquet", repo_type="dataset"))
save("smish_bn", d.assign(label=(d.label != "normal").astype(int)), "smishing+promo")
d = pd.read_csv(h("MOZNLP/MOZ-Smishing", "test.csv", repo_type="dataset"))
save("smish_pt", bal(d.assign(label=(d.label == "Smishing").astype(int)), 1000), "smishing")
d = pd.read_csv(h("jmjmjm3/kor-smishing-message", "smishing.csv", repo_type="dataset"))
save("smish_ko", bal(d.rename(columns={"content": "text"}).assign(label=(d.label == "스미싱").astype(int)), 1000), "smishing")
