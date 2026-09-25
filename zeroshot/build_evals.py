"""Zero-shot classification benchmark (eval only, never trained on).  .venv/bin/python zeroshot/build_evals.py OUTDIR

Classification sets -> OUTDIR/<name>.json: {"template": str, "labels": [...], "items": [{"text", "gold", "lang"}]}
NLI set (XNLI test, binary entailment) -> OUTDIR/xnli.json: {"type": "nli", "items": [{"premise", "hypothesis", "label", "lang"}]}
Hypotheses are English for every language (the transformers pipeline default).
"""
import gzip, json, os, random, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
dl = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")
LANGS = ["en", "de", "fr", "es", "pt", "ru", "pl", "tr", "ar", "hi", "zh", "ja", "ko", "vi", "id", "sw"]


def save(name, template, labels, items):
    json.dump(dict(template=template, labels=labels, items=items), open(f"{OUT}/{name}.json", "w"), ensure_ascii=False)
    print(name, len(labels), "labels", len(items), "items", flush=True)


def sample(rows, n):
    rows = list(rows); rng.shuffle(rows); return rows[:n]


# MASSIVE intents (CC-BY-4.0; 60 intents), 300 per language
items, labels = [], None
for l in LANGS:
    f = dl("mteb/amazon_massive_intent", f"test/{'zh-CN' if l == 'zh' else l}.json.gz")
    rows = [json.loads(x) for x in gzip.open(f, "rt")]
    if labels is None:
        labels = sorted({r["label"] for r in rows})
    items += [dict(text=r["text"], gold=labels.index(r["label"]), lang=l) for r in sample(rows, 300)]
save("massive", "This request is about {}.", [x.replace("_", " ") for x in labels], items)

# SIB-200 topics (CC-BY-SA-4.0; 7 topics), full test (204) per language
SIB = dict(en="eng_Latn", de="deu_Latn", fr="fra_Latn", es="spa_Latn", pt="por_Latn", ru="rus_Cyrl", pl="pol_Latn",
           tr="tur_Latn", ar="arb_Arab", hi="hin_Deva", zh="zho_Hans", ja="jpn_Jpan", ko="kor_Hang", vi="vie_Latn",
           id="ind_Latn", sw="swh_Latn")
SL = ["entertainment", "geography", "health", "politics", "science/technology", "sports", "travel"]
items = []
for l, code in SIB.items():
    df = pd.read_csv(dl("Davlan/sib200", f"data/{code}/test.tsv"), sep="\t", quoting=3)
    items += [dict(text=t, gold=SL.index(c), lang=l) for t, c in zip(df.text, df.category)]
save("sib200", "This text is about {}.", [x.replace("/", " and ") for x in SL], items)

# XNLI test (CC-BY-NC-4.0, evaluation only): entailment (0) vs not
items = []
for l in ["en", "de", "fr", "es", "ru", "tr", "ar", "hi", "zh", "vi", "sw", "th"]:
    df = pd.read_parquet(dl("facebook/xnli", f"{l}/test-00000-of-00001.parquet"))
    items += [dict(premise=r.premise, hypothesis=r.hypothesis, label=int(r.label == 0), lang=l) for r in sample(df.itertuples(), 600)]
json.dump(dict(type="nli", items=items), open(f"{OUT}/xnli.json", "w"), ensure_ascii=False)
print("xnli", len(items))

# English classics, 1000 each
df = pd.read_parquet(dl("fancyzhx/ag_news", "data/test-00000-of-00001.parquet"))
save("agnews", "This news article is about {}.", ["world news", "sports", "business", "science and technology"],
     [dict(text=r.text, gold=int(r.label), lang="en") for r in sample(df.itertuples(), 1000)])
df = pd.read_parquet(dl("community-datasets/yahoo_answers_topics", "yahoo_answers_topics/test-00000-of-00001.parquet"))
Y = ["society and culture", "science and mathematics", "health", "education and reference", "computers and internet",
     "sports", "business and finance", "entertainment and music", "family and relationships", "politics and government"]
save("yahoo", "This question is about {}.", Y,
     [dict(text=(r.question_title + " " + (r.question_content or "")).strip(), gold=int(r.topic), lang="en") for r in sample(df.itertuples(), 1000)])
df = pd.read_parquet(dl("dair-ai/emotion", "split/test-00000-of-00001.parquet"))
save("emotion", "This text expresses {}.", ["sadness", "joy", "love", "anger", "fear", "surprise"],
     [dict(text=r.text, gold=int(r.label), lang="en") for r in sample(df.itertuples(), 1000)])
df = pd.read_parquet(dl("stanfordnlp/sst2", "data/validation-00000-of-00001.parquet"))
save("sst2", "The sentiment of this review is {}.", ["negative", "positive"],
     [dict(text=r.sentence, gold=int(r.label), lang="en") for r in sample(df.itertuples(), 1000)])
df = pd.read_parquet(dl("mteb/banking77", "data/test-00000-of-00001.parquet"))
B = sorted(df.label_text.unique())
save("banking77", "This customer message is about {}.", [x.replace("_", " ") for x in B],
     [dict(text=r.text, gold=B.index(r.label_text), lang="en") for r in sample(df.itertuples(), 1000)])
