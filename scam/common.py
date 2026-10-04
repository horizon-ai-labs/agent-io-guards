"""Shared helpers for spam / phishing / scam evaluation (no sklearn)."""
import glob, os
import numpy as np, pandas as pd

FAMILIES = {"email": ["phish_email", "enron_spam"], "sms_en": ["sms_en"], "smish": ["smish_bn", "smish_pt", "smish_ko"]}


def load_evals(d):
    return {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{d}/*.parquet"))}


def auc(y, s):
    y = np.asarray(y); s = np.asarray(s, float); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    for v in np.unique(s):   # average ranks for ties
        m = s == v
        if m.sum() > 1:
            r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


def score(df, p_unsafe):
    y = df.label.values.astype(int); p = np.asarray(p_unsafe); pr = (p >= 0.5).astype(int)
    tp = int(((pr == 1) & (y == 1)).sum()); fp = int(((pr == 1) & (y == 0)).sum()); fn = int(((pr == 0) & (y == 1)).sum())
    return dict(n=len(y), auc=auc(y, p), f1=2 * tp / max(1, 2 * tp + fp + fn), fpr=fp / max(1, (y == 0).sum()), recall=tp / max(1, (y == 1).sum()))


def summarize(r):
    out = {}
    for fam, sets in FAMILIES.items():
        v = [r[s] for s in sets if s in r]
        if v:
            out[f"_{fam}_auc"] = float(np.mean([x["auc"] for x in v])); out[f"_{fam}_f1"] = float(np.mean([x["f1"] for x in v]))
    m = [x for k, x in r.items() if k.startswith("sms_") and k != "sms_en" and isinstance(x, dict)]
    if m:
        out["_sms_multi_auc"] = float(np.mean([x["auc"] for x in m])); out["_sms_multi_f1"] = float(np.mean([x["f1"] for x in m]))
    return out
