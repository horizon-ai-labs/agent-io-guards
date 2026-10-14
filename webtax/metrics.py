"""Topic / format accuracy and macro-F1 of predicted label slugs against a reference; per-language mean accuracy."""
import numpy as np


def acc_f1(p, r, lang=None):
    p, r = np.asarray(p), np.asarray(r); out = dict(n=len(r), acc=float((p == r).mean()))
    f1 = []
    for c in np.unique(r):
        if (r == c).sum() >= 10:
            tp = ((p == c) & (r == c)).sum(); f1.append(2 * tp / max(1, 2 * tp + ((p == c) & (r != c)).sum() + ((p != c) & (r == c)).sum()))
    out["macro_f1"] = float(np.mean(f1))
    if lang is not None:
        lang = np.asarray(lang); out["lang_mean_acc"] = float(np.mean([(p[lang == l] == r[lang == l]).mean() for l in np.unique(lang)]))
    return out
