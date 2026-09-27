"""Content-safety data v4 = v3 + machine-translated items (safety/translate_prep.py -> translate_items.py -> label_teacher.py).

python safety/build_safety_v4.py V3_DIR TR_LABELLED EVAL_DIR OUT
Translated Civil Comments keep their annotator toxicity as soft target and their subtype categories; translated pool
prompts get the teacher score on the translated text (P(unsafe) + 0.5 P(controversial)), category loss masked.
Rows matching a benchmark text are removed.
"""
import glob, os, re, shutil, sys
import numpy as np, pandas as pd

V3, TR, EV, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
bench = set()
for p in glob.glob(f"{EV}/*.parquet"):
    bench |= {norm(x) for x in pd.read_parquet(p).text}
tr = pd.read_parquet(TR)
tr = tr[~tr.text.map(norm).isin(bench)]
civ = tr.source == "civil"
teach = (tr.p_unsafe + 0.5 * tr.p_contro).clip(0, 1).values
soft = np.where(civ, tr.soft.fillna(0).values.astype(float), teach)
print("translated", len(tr), "civil", int(civ.sum()), "| teacher vs annotator on civil (agree at 0.5):",
      round(float(((teach[civ.values] > 0.5) == (soft[civ.values] > 0.5)).mean()), 3), flush=True)
add = pd.DataFrame(dict(text=tr.text.values, text_pair="", unsafe=(soft > 0.5).astype(int), soft=soft,
                        cats=[list(c) if s == "civil" else [] for c, s in zip(tr.cats, tr.source)], has_cats=civ.values,
                        lang=tr.lang.values, kind="prompt", tag="tr/" + tr.source.str.split("/").str[0].values, id=""))
out = pd.concat([pd.read_parquet(f"{V3}/train.parquet"), add], ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{OUT}/train.parquet")
for f in ["val.parquet", "test.parquet", "stats.json"]:
    shutil.copy(f"{V3}/{f}", f"{OUT}/{f}")
print("train", len(out)); print(add.groupby(["tag", "lang"]).unsafe.mean().unstack(0).round(2).to_string())
