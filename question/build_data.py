"""Multilingual question-vs-statement data. Labels are objective (no LLM): 1 = question, 0 = statement or keyword query
(the label convention of shahrukhx01/question-vs-statement-classifier: LABEL_1 = question).
- sentences: real web sentences from FineWeb-2 / FineWeb (ODC-BY; the edu classifier's samples, edu/prep_fw.py): a sentence ending in a
  question mark (? ？ ؟ ; for Greek) is a question, other sentences are statements. Final punctuation is removed from the input
  (and Spanish ¿ ¡), and half of the texts are lowercased, so the model must decide from the words, as with ASR output or typed queries.
- queries: the reranker's generated search queries (rerank/gen_queries.py; Qwen3.8-27B): type 'question' (final ? removed) vs 'keywords'.
Train from the edu TRAIN documents and query passages pid % 10 != 0; test from the edu TEST documents and pid % 10 == 0 (disjoint).
python question/build_data.py TRAIN_DOCS TEST_DOCS RR_QUERY_DIR OUT_DIR EVAL_DIR"""
import glob, json, os, random, re, sys
import numpy as np, pandas as pd
tr_docs, te_docs, qdir, out, ev = sys.argv[1:6]
QM = re.compile(r"[?？؟]\s*$")
SPLIT = re.compile(r"(?<=[.!?。！？؟।։;\u037e])\s+|(?<=[。！？])")  # incl. Indic danda, Armenian full stop, Greek question mark
rng = random.Random(0)


def strip_end(t):
    t = re.sub(r"[\s.!?。！？؟;\u037e…:،।։]+$", "", t.strip()); t = re.sub(r"^[¿¡]+", "", t)
    return t.lower() if rng.random() < 0.5 else t


def sentences(docs, per_lang_q, per_lang_s):
    rows = []
    for lang, g in docs.groupby("lang"):
        q, s = [], []
        for doc in g.text:
            for sent in SPLIT.split(doc):
                sent = sent.strip()
                if not (15 <= len(sent) <= 300) or "\n" in sent or "http" in sent:
                    continue
                isq = (bool(QM.search(sent)) or (lang == "ell_Grek" and sent.rstrip().endswith((";", "\u037e")))
                       or (lang == "hye_Armn" and "\u055e" in sent))  # Greek ; / U+037E, Armenian ՞ (on the stressed word)
                if lang == "hye_Armn":
                    sent = sent.replace("\u055e", "")
                if lang != "ell_Grek" and sent.rstrip().endswith(";"):
                    continue
                (q if isq else s).append(sent)
        rng.shuffle(q); rng.shuffle(s); n = min(len(q), per_lang_q)
        rows += [dict(text=strip_end(x), label=1, lang=lang, source="sentence") for x in q[:n]]
        rows += [dict(text=strip_end(x), label=0, lang=lang, source="sentence") for x in s[:max(n, min(len(s), per_lang_s))]]
    return pd.DataFrame(rows)


tr = sentences(pd.read_parquet(tr_docs), 3000, 3000); te = sentences(pd.read_parquet(te_docs), 100, 100)
qs = pd.concat([pd.read_json(p, lines=True) for p in sorted(glob.glob(f"{qdir}/q_*.jsonl"))], ignore_index=True)
qs = qs[qs.qtype.isin(["question", "keywords"])].dropna(subset=["query"])
qs = pd.DataFrame(dict(text=[strip_end(t) for t in qs["query"]], label=(qs.qtype == "question").astype(int).values, lang=qs.qlang.values,
                       source="query", test=(qs.pid % 10 == 0).values))
qtr = qs[~qs.test].groupby(["lang", "label"]).head(1200); qte = qs[qs.test].groupby(["lang", "label"]).head(100)
train = pd.concat([tr, qtr.drop(columns="test")], ignore_index=True).drop_duplicates("text")
test_s, test_q = te.drop_duplicates("text"), qte.drop(columns="test").drop_duplicates("text")
train = train[~train.text.isin(set(test_s.text) | set(test_q.text))]
train["val"] = np.random.RandomState(0).rand(len(train)) < 0.05
os.makedirs(out, exist_ok=True); os.makedirs(ev, exist_ok=True)
train[~train.val].drop(columns="val").sample(frac=1, random_state=0).to_parquet(f"{out}/train.parquet"); train[train.val].drop(columns="val").to_parquet(f"{out}/val.parquet")
test_s.to_parquet(f"{ev}/sentences.parquet"); test_q.to_parquet(f"{ev}/queries.parquet")
st = dict(train=int((~train.val).sum()), val=int(train.val.sum()), by_source=train.source.value_counts().to_dict(), n_lang=int(train.lang.nunique()),
          pos_rate=round(float(train.label.mean()), 3), test_sentences=len(test_s), test_sent_langs=int(test_s.lang.nunique()), test_queries=len(test_q),
          test_query_langs=int(test_q.lang.nunique()))
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
