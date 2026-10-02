"""NER training data from Qwen labels.   python ner/build_ner.py OUT LABEL_DIR [LABEL_DIR ...]
Input: ner/teacher_ner.py label output (pid, text, lang, ents [[start, end, type]]). Output (pii/train_tok.py format):
OUT/train.parquet, val.parquet with text, spans_json, source (= "fw_<lang>"). Val = 2% of passages. Passages that contain
an evaluation sentence (ner/evals, normalised) are removed.
"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
OUT, dirs = sys.argv[1], sys.argv[2:]; os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
ev = set()
for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")):
    for toks in pd.read_parquet(p).tokens:
        s = norm(" ".join(toks))
        if len(s) >= 40:
            ev.add(s)
rows = [json.loads(l) for d in dirs for f in sorted(glob.glob(f"{d}/*.jsonl")) for l in open(f)]
df = pd.DataFrame(rows).drop_duplicates("pid")
SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")
bad = df.text.map(lambda t: any(norm(x) in ev for x in SPLIT.split(t)))   # exact match of a passage sentence with an eval sentence
st = dict(n_in=len(df), removed_eval_overlap=int(bad.sum())); df = df[~bad]
df["spans_json"] = df.ents.map(json.dumps); df["source"] = "fw_" + df.lang
rng = np.random.RandomState(0); isv = rng.rand(len(df)) < 0.02
df[~isv][["text", "spans_json", "source"]].to_parquet(f"{OUT}/train.parquet"); df[isv][["text", "spans_json", "source"]].to_parquet(f"{OUT}/val.parquet")
ty = pd.Series([e[2] for es in df.ents for e in es]).value_counts().to_dict()
st.update(n_train=int((~isv).sum()), n_val=int(isv.sum()), entities=ty, passages_without_entities=int((df.ents.map(len) == 0).sum()), n_lang=int(df.lang.nunique()))
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1); print(json.dumps(st))
