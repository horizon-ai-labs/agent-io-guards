"""Training windows from Qwen spoken-style transcripts (punct/gen_spoken.py) + merged training dir.
python punct/build_spoken.py OUT_DIR EVAL_DIR V1_DATA_DIR SPOKEN.jsonl [...]
Windows as in build_data.py (<= 230 units, 30% cut mid-sentence); transcripts sharing a sentence with any eval set are dropped.
OUT_DIR/spoken.parquet and OUT_DIR/data/{train,val}.parquet (= v1 train + spoken windows; v1 val)."""
import json, os, random, re, shutil, sys
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import units, L2I, NOSPACE

OUT, EV, V1 = sys.argv[1:4]; ins = sys.argv[4:]
os.makedirs(f"{OUT}/data", exist_ok=True); rng = random.Random(0)
CODE = {"Chinese (Simplified)": "cmn_Hani", "Chinese (Traditional)": "cmn_Hani", "Japanese": "jpn_Jpan", "Greek": "ell_Grek", "Thai": "tha_Thai"}
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
SPLIT = re.compile(r"(?<=[.!?。！？])\s*")
ev = set()
for f in ["flores.parquet", "ted.parquet", "europarl.parquet"]:
    for t in pd.read_parquet(f"{EV}/{f}").text:
        ev |= {norm(x) for x in SPLIT.split(t) if len(x) > 25}
rows, dropped, n = [], 0, 0
for fn in ins:
    for line in open(fn):
        d = json.loads(line); n += 1
        lang = CODE.get(d["lang"], "xxx_Latn")
        if lang == "tha_Thai":
            continue
        t = d["text"]
        if any(norm(x) in ev for x in SPLIT.split(t) if len(x) > 25):
            dropped += 1; continue
        c, e, l = units(t, lang); nos = lang in NOSPACE; cap = 400 if nos else 230
        for a in range(0, len(e), cap):
            ee, ll = e[a:a + cap], l[a:a + cap]
            if len(ee) < 8:
                continue
            off = e[a - 1] + (0 if nos else 1) if a else 0
            if rng.random() < 0.3 and len(ee) > 12:
                cut = rng.randint(8, len(ee) - 1); ee, ll = ee[:cut], ll[:cut]
            rows.append((c[off:ee[-1]], [x - off for x in ee], [L2I[x] for x in ll], d["lang"]))
sp = pd.DataFrame(rows, columns=["text", "ends", "labs", "lang"]); sp.to_parquet(f"{OUT}/spoken.parquet")
print("transcripts", n, "dropped (eval overlap)", dropped, "windows", len(sp), "languages", sp.lang.nunique(), flush=True)
t = pd.read_parquet(f"{V1}/train.parquet")
pd.concat([t, sp], ignore_index=True).sample(frac=1.0, random_state=0).to_parquet(f"{OUT}/data/train.parquet")
shutil.copy(f"{V1}/val.parquet", f"{OUT}/data/val.parquet"); print("train", len(t) + len(sp))
