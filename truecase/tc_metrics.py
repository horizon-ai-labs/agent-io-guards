"""Shared chunking and scoring for truecase/eval_tc.py and truecase/pcs_preds.py (same chunks for every system)."""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "punct"))
from common import LABELS as PL, NOSPACE, units  # noqa
from common_tc import unit_cases


def unit_labels(offsets, labels, ends, default):
    """per-token label strings -> label per unit (the token that ends at the unit end)."""
    at = {e: l for (b, e), l in zip(offsets, labels) if e > b}
    return [at.get(e, default) for e in ends]


def chunks(ev):
    items = []
    for p in sorted(os.listdir(ev)):
        if not p.endswith(".parquet"):
            continue
        name = p[:-8]; df = pd.read_parquet(f"{ev}/{p}")
        for text, lang in zip(df.text, df.lang):
            nos = lang.startswith(("zho", "jpn", "yue")) or lang in NOSPACE
            c, e, l = units(text, "cmn_Hani" if lang.startswith(("zho", "yue")) else lang); cap = 250 if nos else 150
            for a in range(0, len(e), cap):
                off = e[a - 1] + (0 if nos else 1) if a else 0; ee = [x - off for x in e[a:a + cap]]; ch = c[off:e[a:a + cap][-1]]
                items.append(dict(set=name, lang=lang, text=ch, ends=ee, gp=l[a:a + cap], gc=unit_cases(ch, ee, nos), nos=nos))
    return items


def score(items, pp, pc):
    res = {}
    for it, p, c in zip(items, pp, pc):
        d = res.setdefault(it["set"], {}).setdefault(it["lang"], dict(pconf=np.zeros((6, 6), np.int64), cconf=np.zeros((3, 3), np.int64)))
        for g, q in zip(it["gp"], p):
            d["pconf"][PL.index(g), PL.index(q)] += 1
        for g, q in zip(it["gc"], c):
            d["cconf"][g, q] += 1
    out = {}
    for s, langs in res.items():
        for l, d in langs.items():
            pc_, cc = d["pconf"], d["cconf"]
            tp = sum(pc_[i, i] for i in range(1, 6)); fp = pc_[:, 1:].sum() - tp; fn = pc_[1:, :].sum() - tp
            ctp = cc[1, 1] + cc[2, 2]; cfp = cc[:, 1:].sum() - ctp; cfn = cc[1:, :].sum() - ctp
            out.setdefault(s, {})[l] = dict(punct_f1=float(2 * tp / max(1, 2 * tp + fp + fn)), case_acc=float(np.trace(cc) / max(1, cc.sum())),
                                            case_f1=float(2 * ctp / max(1, 2 * ctp + cfp + cfn)) if cc[1:, :].sum() > 0 else None, n_words=int(cc.sum()))
    summ = {}
    for s, langs in out.items():
        summ[s] = dict(punct_f1=float(np.mean([v["punct_f1"] for v in langs.values()])),
                       case_f1=float(np.mean([v["case_f1"] for v in langs.values() if v["case_f1"] is not None])),
                       case_acc=float(np.mean([v["case_acc"] for v in langs.values()])), n_langs=len(langs))
    return dict(per_lang=out, summary=summ)


