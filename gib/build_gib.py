"""Multilingual gibberish-detection data (4 classes, the label set of madhurjindal/autonlp-Gibberish-Detector).
python gib/build_gib.py OUT_DIR TEXTS.parquet QUERIES_DIR [PER_LANG] [EXTRA_TEXTS.parquet]
EXTRA_TEXTS (v1): language-ID training text (lid_v5 train.parquet: text, label = FLORES code; FineWeb-2-derived, FLORES
sentences removed) adds the languages missing from TEXTS - v0 flagged unseen languages as noise on FLORES.
- clean: real text: FineWeb-2/FineWeb snippets (1-3 sentences or a short span) and search queries written by Qwen for web
  passages (questions and keyword queries - keyword queries are deliberately CLEAN, unlike in the original detector).
- mild gibberish: a clean sentence with heavy local damage: typos (swap/drop/insert/double characters) in 30-60% of the
  words, 10-25% of words dropped, some neighbouring words swapped. The topic is still recognisable, the sentence is broken.
- word salad: real words of the language in a meaningless order (words pooled from 2-4 unrelated sentences, shuffled,
  3-16 words).
- noise: no real words: random strings over the language's own characters (sampled from its text), keyboard mashing,
  repeated characters, digit/symbol runs, base64/hex-like tokens, mixed scripts.
Splits by language are random (train 96% / val 2% / test 2%); FLORES-200 sentences are kept for evaluation only.
"""
import glob, json, os, random, re, sys, unicodedata
import numpy as np, pandas as pd

OUT, TEXTS, QDIR = sys.argv[1:4]; PER = int(sys.argv[4]) if len(sys.argv) > 4 else 4000
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
SPLIT = re.compile(r"(?<=[.!?。！？।؟])\s+")
NOSPACE = {"jpn_Jpan", "cmn_Hani", "zho_Hans", "zho_Hant", "tha_Thai", "khm_Khmr", "lao_Laoo", "mya_Mymr", "bod_Tibt"}
KEYB = ["qwertyuiop", "asdfghjkl", "zxcvbnm", "1234567890", "йцукенгшщзхъ", "фывапролджэ", "ячсмитьбю"]


def words(t, lang):
    return list(t.replace(" ", "")) if lang in NOSPACE else t.split()


def join(ws, lang):
    return "".join(ws) if lang in NOSPACE else " ".join(ws)


def typo(w, chars):
    if len(w) < 2:
        return w
    i = rng.randrange(len(w)); op = rng.random()
    if op < 0.3 and len(w) > 2:
        return w[:i] + w[i + 1:]
    if op < 0.55 and i < len(w) - 1:
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    if op < 0.8:
        return w[:i] + rng.choice(chars) + w[i:]
    return w[:i] + w[i] * 2 + w[i + 1:]


def mild(t, lang, chars):
    ws = words(t, lang)
    if len(ws) < 4:
        return None
    p = rng.uniform(0.3, 0.6)
    ws = [typo(w, chars) if rng.random() < p else w for w in ws]
    ws = [w for w in ws if rng.random() > rng.uniform(0.1, 0.25)] or ws
    for _ in range(max(1, len(ws) // 6)):
        i = rng.randrange(max(1, len(ws) - 1))
        if i + 1 < len(ws):
            ws[i], ws[i + 1] = ws[i + 1], ws[i]
    return join(ws, lang)


def salad(pool, lang):
    ws = [w for s in rng.sample(pool, min(len(pool), rng.randint(2, 4))) for w in words(s, lang)]
    rng.shuffle(ws)
    n = rng.randint(3, 16) if lang not in NOSPACE else rng.randint(6, 30)
    return join(ws[:n], lang) if len(ws) >= 3 else None


def noise(chars, lang):
    kind = rng.random()
    if kind < 0.45:   # random strings over the language's characters
        ws = ["".join(rng.choice(chars) for _ in range(rng.randint(2, 12))) for _ in range(rng.randint(1, 10))]
        return join(ws, lang)
    if kind < 0.65:   # keyboard mashing
        row = rng.choice(KEYB); return " ".join("".join(rng.choice(row) for _ in range(rng.randint(3, 12))) for _ in range(rng.randint(1, 5)))
    if kind < 0.75:   # repeated characters
        return " ".join(rng.choice(chars) * rng.randint(3, 15) for _ in range(rng.randint(1, 4)))
    if kind < 0.85:   # digits / symbols
        return "".join(rng.choice("0123456789!@#$%^&*()_+-=[]{};:,./?~ ") for _ in range(rng.randint(5, 40)))
    if kind < 0.93:   # base64/hex-like
        a = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/" if rng.random() < 0.5 else "0123456789abcdef"
        return " ".join("".join(rng.choice(a) for _ in range(rng.randint(8, 32))) for _ in range(rng.randint(1, 3)))
    other = rng.choice(list(SCRIPTS.values()))   # mixed scripts
    return " ".join("".join(rng.choice(rng.choice([chars, other])) for _ in range(rng.randint(2, 10))) for _ in range(rng.randint(2, 8)))


tx = pd.read_parquet(TEXTS, columns=["text", "lang"])
if len(sys.argv) > 5:
    ex = pd.read_parquet(sys.argv[5], columns=["text", "label"]).rename(columns={"label": "lang"})
    ex = ex[~ex.lang.isin(set(tx.lang))]   # only languages TEXTS lacks
    tx = pd.concat([tx, ex], ignore_index=True); print("extra languages", ex.lang.nunique(), flush=True)
q = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(f"{QDIR}/*.jsonl"))], ignore_index=True)
plang = dict(pd.read_parquet(os.path.join(os.path.dirname(TEXTS), "rr", "passages.parquet"), columns=["pid", "lang"]).values) \
    if os.path.exists(os.path.join(os.path.dirname(TEXTS), "rr", "passages.parquet")) else {}
q["lang"] = np.where(q.qtype == "english", "eng_Latn", q.pid.map(plang))
SCRIPTS = {}
rows = []
for lang, g in tx.groupby("lang"):
    texts = g.text.sample(min(len(g), PER * 3), random_state=0).tolist()
    sents = [s.strip() for t in texts for s in SPLIT.split(t) if 10 <= len(s.strip()) <= 300]
    if len(sents) < 100:
        continue
    cnt = {}
    for s in sents[:3000]:
        for c in s:
            if c.isalpha():
                cnt[c] = cnt.get(c, 0) + 1
    chars = [c for c, _ in sorted(cnt.items(), key=lambda x: -x[1])[:80]]
    SCRIPTS[lang] = chars
    rng.shuffle(sents)
    n = PER // 4
    clean = sents[:n // 2]
    clean += [t[:rng.randint(15, 120)].rsplit(" ", 1)[0] if " " in t[:120] else t[:60] for t in texts[:n // 4]]   # short spans
    ql = q[q.lang == lang]["query"].tolist(); rng.shuffle(ql); clean += ql[:n - len(clean)]
    rows += [(t, "clean", lang) for t in clean[:n]]
    src = sents[n // 2:]
    m = [x for x in (mild(s, lang, chars) for s in src[:n * 2]) if x][:n]; rows += [(t, "mild gibberish", lang) for t in m]
    sw = [x for x in (salad(src, lang) for _ in range(n * 2)) if x][:n]; rows += [(t, "word salad", lang) for t in sw]
    rows += [(noise(chars, lang), "noise", lang) for _ in range(n)]
df = pd.DataFrame(rows, columns=["text", "label", "lang"]).drop_duplicates("text").sample(frac=1.0, random_state=0).reset_index(drop=True)
df = df[df.text.str.strip().str.len() >= 2]
u = np.random.RandomState(1).rand(len(df))
df[u < 0.96].to_parquet(f"{OUT}/train.parquet"); df[(u >= 0.96) & (u < 0.98)].to_parquet(f"{OUT}/val.parquet"); df[u >= 0.98].to_parquet(f"{OUT}/test.parquet")
json.dump(dict(n=len(df), by_label=df.label.value_counts().to_dict(), n_lang=int(df.lang.nunique())), open(f"{OUT}/stats.json", "w"), indent=1)
print(len(df), df.label.value_counts().to_dict(), df.lang.nunique())
for lab in ["clean", "mild gibberish", "word salad", "noise"]:
    for t in df[(df.label == lab) & (df.lang.isin(["eng_Latn", "deu_Latn", "jpn_Jpan"]))].text.head(2):
        print(lab, "|", t[:110])
