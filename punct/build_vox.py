"""Spoken-style punctuation data from VoxPopuli (CC0 transcriptions of European Parliament speeches; 'original_text' = the
edited verbatim report aligned to the audio).   python punct/build_vox.py OUT_DIR EVAL_DIR [MAX_WINDOWS_PER_LANG]
Paragraphs (segments of one paragraph_id, in time order) -> windows with punct/common.py units, like build_data.py.
Removes any paragraph that shares a sentence with the Europarl or TED eval sets (normalised). Output OUT_DIR/vox.parquet."""
import gzip, io, os, random, re, sys, urllib.request
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import units, L2I

OUT, EV = sys.argv[1:3]; MAXW = int(sys.argv[3]) if len(sys.argv) > 3 else 15000
os.makedirs(OUT, exist_ok=True); rng = random.Random(0)
LANGS = dict(en="eng_Latn", de="deu_Latn", fr="fra_Latn", es="spa_Latn", pl="pol_Latn", it="ita_Latn", ro="ron_Latn", hu="hun_Latn",
             cs="ces_Latn", nl="nld_Latn", fi="fin_Latn", hr="hrv_Latn", sk="slk_Latn", sl="slv_Latn", et="ekk_Latn", lt="lit_Latn")
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
SPLIT = re.compile(r"(?<=[.!?])\s+")
ev = set()
for f in ["europarl.parquet", "ted.parquet"]:
    for t in pd.read_parquet(f"{EV}/{f}").text:
        ev |= {norm(x) for x in SPLIT.split(t) if len(x) > 25}
rows = []
for code, lang in LANGS.items():
    for k in range(5):
        try:
            raw = urllib.request.urlopen(f"https://dl.fbaipublicfiles.com/voxpopuli/annotations/asr/asr_{code}.tsv.gz", timeout=600).read(); break
        except Exception as e:
            print("retry", code, e, flush=True)
    df = pd.read_csv(io.BytesIO(gzip.decompress(raw)), sep="|", usecols=["id_", "paragraph_id", "original_text"], quoting=3, dtype=str).dropna()
    paras = df.sort_values("id_").groupby("paragraph_id").original_text.apply(lambda x: " ".join(s.strip() for s in x)).tolist()
    rng.shuffle(paras); n0 = len(rows); dropped = 0
    for p in paras:
        if any(norm(x) in ev for x in SPLIT.split(p) if len(x) > 25):
            dropped += 1; continue
        c, e, l = units(p, lang)
        for a in range(0, len(e), 230):
            ee, ll = e[a:a + 230], l[a:a + 230]
            if len(ee) < 8:
                continue
            off = e[a - 1] + 1 if a else 0
            if rng.random() < 0.3 and len(ee) > 12:
                cut = rng.randint(8, len(ee) - 1); ee, ll = ee[:cut], ll[:cut]
            rows.append((c[off:ee[-1]], [x - off for x in ee], [L2I[x] for x in ll], lang))
        if len(rows) - n0 >= MAXW:
            break
    print(code, "paragraphs", len(paras), "dropped (eval overlap)", dropped, "windows", len(rows) - n0, flush=True)
pd.DataFrame(rows, columns=["text", "ends", "labs", "lang"]).to_parquet(f"{OUT}/vox.parquet"); print("total", len(rows))
