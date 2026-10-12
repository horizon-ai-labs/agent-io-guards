import numpy as np, pandas as pd


def auc(y, s):
    y, s = np.asarray(y).astype(bool), np.asarray(s, float)
    if y.all() or not y.any():
        return None
    r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1(y, p):
    y, p = np.asarray(y).astype(bool), np.asarray(p).astype(bool); tp = (y & p).sum()
    return 0.0 if tp == 0 else float(2 * tp / (2 * tp + (~y & p).sum() + (y & ~p).sum()))
