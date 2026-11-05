"""Coherent passages for compression training: FineWeb-2 documents (ODC-BY; edu/prep_fw.py docs in .gated/edu/label_in.parquet, 65
languages) and the spoken-style transcripts written by Qwen3.8-27B for punctuation v1.1 (Apache-2.0, 69 languages). A passage = the
start of a document cut at paragraph (or sentence) boundaries, 60-350 words (120-600 characters per CJK passage). One passage per doc.
Excludes unspaced scripts (SKIP) and passages sharing a sentence with the QA eval contexts.
python compress/build_passages.py OUT.parquet PER_LANG_FW PER_LANG_SPOKEN SEED SPOKEN_JSONL[,...] [EXCLUDE_IDS.parquet]"""
import json, re, sys
import pandas as pd
sys.path.insert(0, __file__.rsplit("/", 1)[0]); from align import nospace, code
SKIP = ("tha_", "lao_", "khm_", "mya_", "bod_")   # unspaced scripts without character-level meaning: whitespace units are whole phrases
SPLIT = re.compile(r"(?<=[.!?。！？])\s*|\n+")
nrm = lambda x: re.sub(r"\W+", " ", x.lower()).strip()
EVAL = {nrm(x) for c in pd.read_parquet("compress/evals/qa.parquet").context for x in SPLIT.split(c) if len(x) > 40}
overlap = lambda t: any(nrm(x) in EVAL for x in SPLIT.split(t) if len(x) > 40)

out, n_fw, n_sp, seed, sp_files = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), sys.argv[5]
excl = set(pd.read_parquet(sys.argv[6]).src_id) if len(sys.argv) > 6 else set()


def passage(text, lang):
    ns = nospace(lang); size = (lambda s: len(s.replace(" ", ""))) if ns else (lambda s: len(s.split()))
    lo, hi = (120, 600) if ns else (60, 350)
    parts = [p.strip() for p in re.split(r"\n\s*\n|\n", text) if p.strip()]
    out = []
    for p in parts:
        if size(" ".join(out + [p])) > hi:
            if size(" ".join(out)) >= lo:
                break
            sents = re.split(r"(?<=[.!?。！？])\s*", p)
            for s in sents:
                if size(" ".join(out + [s])) > hi:
                    break
                out.append(s)
            break
        out.append(p)
    t = "\n".join(out).strip()
    return t if size(t) >= lo else None


rows = []
fw = pd.read_parquet(".gated/edu/label_in.parquet", columns=["text", "lang", "source"])
fw = fw.reset_index(drop=True)   # source fineweb2 (65 langs) + llama_calib (English FineWeb docs), both ODC-BY
fw["src_id"] = "fw:" + fw.index.astype(str)
fw = fw[~fw.src_id.isin(excl)]
fw = fw[~fw.lang.str.startswith(SKIP)]
for lang, g in fw.groupby("lang"):
    g = g.sample(frac=1.0, random_state=seed); k = 0
    for t, sid in zip(g.text, g.src_id):
        p = passage(t, lang)
        if p and not overlap(p):
            rows.append(dict(text=p, lang=lang, src="fineweb", src_id=sid)); k += 1
            if k >= n_fw:
                break
sp = []
for f in sp_files.split(","):
    for i, l in enumerate(open(f)):
        r = json.loads(l); sp.append(dict(text=r["text"], lang=r["lang"], src_id=f"sp:{f.rsplit('/', 1)[-1]}:{i}"))
sp = pd.DataFrame(sp); sp["lang"] = sp.lang.map(code); sp = sp[~sp.src_id.isin(excl) & ~sp.lang.str.startswith(SKIP)]
for lang, g in sp.groupby("lang"):
    g = g.sample(frac=1.0, random_state=seed); k = 0
    for t, sid in zip(g.text, g.src_id):
        p = passage(t, lang)
        if p and not overlap(p):
            rows.append(dict(text=p, lang=lang, src="spoken", src_id=sid)); k += 1
            if k >= n_sp:
                break
d = pd.DataFrame(rows); d["id"] = range(len(d)); d.to_parquet(out, index=False)
print(len(d), d.groupby("src").lang.nunique().to_dict(), d.groupby("src").size().to_dict())
