"""Training windows for punctuation restoration from FineWeb-2 / FineWeb (ODC-BY) snippets (.gated/emb_texts.parquet,
built by emb/build_texts.py: pid, text, lang, url).   python punct/build_data.py TEXTS.parquet OUT_DIR [MAX_PER_LANG]
Filter: the snippet ends with a sentence-final mark, >= 6 units, mostly letters, no 60-unit run without any mark, at most
35% of units labelled (lists). Windows: 1-4 snippets of the same language joined (up to 230 words / 400 CJK characters,
the chunk size of deepmultilingualpunctuation); 30% are cut at a random unit, so a window does not always end in ".".
Output OUT_DIR/{train,val}.parquet: text (clean input), ends (unit end offsets), labs (label ids), lang. Val = 1.5% of
snippets (by pid hash). Lowercasing is applied at training time.
"""
import os, random, sys, unicodedata, zlib
import numpy as np, pandas as pd, pyarrow.parquet as pq
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import units, L2I, NOSPACE, SKIP

SRC, OUT = sys.argv[1:3]; MAXL = int(sys.argv[3]) if len(sys.argv) > 3 else 0
os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
END = tuple(".!?。！？।॥۔።။։؟…")


def ok(t, lang):
    s = t.rstrip("\"'»”’)] ")
    if not s.endswith(END) or len(t) < 30:
        return None
    letters = sum(c.isalpha() or unicodedata.category(c)[0] == "M" for c in t)
    if letters < 0.6 * len(t.replace(" ", "")):
        return None
    c, e, l = units(t, lang)
    if len(e) < 6:
        return None
    nz = [i for i, x in enumerate(l) if x != "0"]
    if len(nz) > 0.35 * len(l) or max(np.diff([-1] + nz + [len(l)])) > 60:
        return None
    return t


pf = pq.ParquetFile(SRC)
by = {}
for g in range(pf.num_row_groups):
    df = pf.read_row_group(g, columns=["pid", "text", "lang"]).to_pandas()
    for pid, t, lang in df.itertuples(index=False):
        if lang in SKIP:
            continue
        by.setdefault(lang, []).append((pid, t))
print("languages", len(by), flush=True)
rows = {"train": [], "val": []}
for lang, items in sorted(by.items()):
    rng.shuffle(items)
    if MAXL:
        items = items[:MAXL * 3]
    keep = {"train": [], "val": []}
    for pid, t in items:
        t = ok(t, lang)
        if t:
            keep["val" if zlib.crc32(str(pid).encode()) % 1000 < 15 else "train"].append(t)
    for split, ts in keep.items():
        cap = 400 if lang in NOSPACE else 230; sep = "" if lang in NOSPACE else " "; i = 0; n0 = len(rows[split])
        while i < len(ts):
            k = 1 if rng.random() < 0.4 else rng.randint(2, 4)
            c, e, l = units(sep.join(ts[i:i + k]), lang); i += k
            for a in range(0, len(e), cap):   # split very long joins into chunks
                ee, ll = e[a:a + cap], l[a:a + cap]
                off = e[a - 1] + (0 if lang in NOSPACE else 1) if a else 0
                if rng.random() < 0.3 and len(ee) > 12:
                    cut = rng.randint(8, len(ee) - 1); ee, ll = ee[:cut], ll[:cut]
                if len(ee) < 4:
                    continue
                txt = c[off:ee[-1]]
                rows[split].append((txt, [x - off for x in ee], [L2I[x] for x in ll], lang))
            if MAXL and len(rows[split]) - n0 >= (MAXL if split == "train" else max(50, MAXL // 50)):
                break
    print(lang, len(items), "kept", len(keep["train"]), "windows", sum(1 for r in rows["train"] if r[3] == lang), flush=True)
for split, r in rows.items():
    df = pd.DataFrame(r, columns=["text", "ends", "labs", "lang"]).sample(frac=1.0, random_state=0)
    df.to_parquet(f"{OUT}/{split}.parquet")
    lab = np.bincount(np.concatenate(df.labs.map(np.array).values), minlength=6)
    print(split, len(df), "units", lab.sum(), "label share", (lab / lab.sum()).round(4).tolist(), flush=True)
