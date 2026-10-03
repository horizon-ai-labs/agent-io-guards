"""Evaluate 1-800-BAD-CODE punctuation models (punctuators package, ONNX; input = lowercased text without punctuation;
they also true-case and split sentences) on the same chunks and metric as punct/eval_punct.py (lowercased input only).
Their output is re-normalised with punct/common.units and aligned to the input words (case-insensitive).
python punct/eval_pcs.py OUT.json EVAL_DIR MODEL [MODEL ...]   (run in a private venv with punctuators installed)
"""
import difflib, json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, NOSPACE, units, lower_keep_len
from punctuators.models import PunctCapSegModelONNX

out_f, ev = sys.argv[1:3]; models = sys.argv[3:]
KREDOR = {"eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "bul_Cyrl", "ita_Latn", "pol_Latn", "nld_Latn", "ces_Latn", "por_Latn",
          "slk_Latn", "slv_Latn"}
items = []
for p in sorted(os.listdir(ev)):
    if not p.endswith(".parquet"):
        continue
    name, df = p[:-8], pd.read_parquet(f"{ev}/{p}")
    for text, lang in zip(df.text, df.lang):
        nos = lang.startswith(("zho", "jpn", "yue")) or lang in NOSPACE
        ul = "cmn_Hani" if lang.startswith(("zho", "yue")) else lang
        c, e, l = units(text, ul)
        cap = 250 if nos else 150
        for a in range(0, len(e), cap):
            off = e[a - 1] + (0 if nos else 1) if a else 0
            ee = e[a:a + cap]
            items.append((name, lang, ul, c[off:ee[-1]], [x - off for x in ee], l[a:a + cap], nos))
print("chunks", len(items), flush=True)
res = json.load(open(out_f)) if os.path.exists(out_f) else {}
for mname in models:
    m = PunctCapSegModelONNX.from_pretrained(mname); t0 = time.time()
    texts = [lower_keep_len(it[3]) for it in items]; outs = []
    for s in range(0, len(texts), 64):
        outs += m.infer(texts[s:s + 64]); print(s, f"{time.time() - t0:.0f}s", flush=True) if s % 6400 == 0 else None
    cnt, miss = {}, 0
    for it, o in zip(items, outs):
        name, lang, ul, clean, ends, gold, nos = it
        oc, oe, ol = units(("" if nos else " ").join(o), ul)
        gw = [clean[(ends[i - 1] if i else 0):x].strip().lower() for i, x in enumerate(ends)]
        pw = [oc[(oe[i - 1] if i else 0):x].strip().lower() for i, x in enumerate(oe)]
        pred = ["0"] * len(gw)
        for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, gw, pw, autojunk=False).get_opcodes():
            if tag == "equal":
                pred[a1:a2] = ol[b1:b2]
            else:
                miss += a2 - a1
        c = cnt.setdefault((name, lang), np.zeros((6, 6), np.int64))
        for g, q in zip(gold, pred):
            c[LABELS.index(g), LABELS.index(q)] += 1
    r = {}
    for (sname, lang), c in cnt.items():
        tp = sum(c[i, i] for i in range(1, 6)); fp = c[:, 1:].sum() - tp; fn = c[1:, :].sum() - tp
        r.setdefault(sname, {})[lang] = {"lower": dict(f1=2 * tp / max(1, 2 * tp + fp + fn), conf=c.tolist(),
                                                     **{LABELS[i]: 2 * c[i, i] / max(1, c[i, :].sum() + c[:, i].sum()) for i in range(1, 6)})}
    summ = {"_unaligned_words": miss}
    for sname, langs in r.items():
        for sub, ls in [("all", list(langs)), ("kredor12", [l for l in langs if l in KREDOR])]:
            if ls:
                summ[f"{sname}/lower/{sub}"] = float(np.mean([langs[l]["lower"]["f1"] for l in ls]))
        pooled = sum(np.array(langs[l]["lower"]["conf"]) for l in langs)
        for i in (1, 2, 3):
            summ[f"{sname}/lower/pooled_{LABELS[i]}"] = float(2 * pooled[i, i] / max(1, pooled[i, :].sum() + pooled[:, i].sum()))
    r["_summary"] = summ; res[mname] = r
    print(mname, f"{time.time() - t0:.0f}s", json.dumps({k: round(v, 4) for k, v in summ.items()}), flush=True)
    json.dump(res, open(out_f, "w"))
