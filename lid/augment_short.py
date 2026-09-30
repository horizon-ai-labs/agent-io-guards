"""Language-ID data v3 = v2 (cleaned) + one extra 10-40 character span from every long training sample.
python lid/augment_short.py V2_DIR OUT
Why: v2 matched GlotLID on 20-40 character prefixes in accuracy (.815 vs .810) but trailed in macro-F1 (.808 vs .819);
short strings (chat messages, queries, titles) are a main use case. Spans are cut at word boundaries for space-separated
scripts; val and eval are unchanged.
v4 (env FILTER_SPANS=1, default): the new spans and the existing samples pass the same script and English filters as
lid/clean_lid.py - in v3, English fragments cut out of code-switched long samples (e.g. Tagalog, Xhosa) were added under
those labels, and short English phrases got low confidence ("Thank you very much for your help" -> eng 0.42).
"""
import os, random, shutil, sys
import pandas as pd

V2, OUT = sys.argv[1:3]
os.makedirs(OUT, exist_ok=True)
rng = random.Random(3)
tr = pd.read_parquet(f"{V2}/train.parquet")


def span(t):
    L = rng.randint(10, 40)
    if len(t) <= L:
        return None
    st = rng.randrange(0, len(t) - L + 1); s = t[st:st + L]
    if " " in s.strip():
        s = s[s.find(" ") + 1:] if st > 0 else s
        s = s[:s.rfind(" ")] if s.rfind(" ") > 4 else s
    s = s.strip()
    return s if len(s) >= 5 else None


FILTER = os.environ.get("FILTER_SPANS", "1") == "1"
if FILTER:
    src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "clean_lid.py")).read()
    ns = {"os": os}; exec(src[src.index("import re, unicodedata"): src.index("ev = pd.read_parquet")], ns)
    ok = lambda s, l: not (l.split("_")[-1] != "Latn" and ns["script_frac"](s, l.split("_")[-1]) < 0.5) and \
        not (l.split("_")[-1] == "Latn" and l != "eng_Latn" and ns["english_like"](s))
    before = len(tr)
    tr = tr[[ok(t, l) for t, l in zip(tr.text, tr.label)]]   # also re-check the existing short spans
    print("existing samples removed by filters", before - len(tr), flush=True)
longs = tr[tr.kind == "long"]
extra = pd.DataFrame([(s, l, "short") for s, l in ((span(t), l) for t, l in zip(longs.text, longs.label)) if s and (not FILTER or ok(s, l))],
                     columns=["text", "label", "kind"])
out = pd.concat([tr, extra]).drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{OUT}/train.parquet")
for f in ["val.parquet", "labels.json"]:
    shutil.copy(f"{V2}/{f}", f"{OUT}/{f}")
shutil.copytree(f"{V2}/eval", f"{OUT}/eval", dirs_exist_ok=True)
print("train", len(tr), "->", len(out), "| short share", round((out.kind == "short").mean(), 3), flush=True)
