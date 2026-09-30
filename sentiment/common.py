"""Shared sentiment helpers (no sklearn: also used in the vLLM venv)."""
import glob, os
import numpy as np, pandas as pd

C3 = ["negative", "neutral", "positive"]


def load_evals(d):
    return {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{d}/*.parquet"))}


def macro_f1(y, p, labels):
    f = []
    for c in labels:
        tp = np.sum((p == c) & (y == c)); fp = np.sum((p == c) & (y != c)); fn = np.sum((p != c) & (y == c))
        f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + fp + fn))
    return float(np.mean(f))


def score(df, probs):
    """probs: (n, 3) over C3. Binary sets (no neutral gold label): argmax over negative/positive only."""
    probs = np.asarray(probs, dtype=float); y = df.label.values
    if int(df.binary.iloc[0]) == 1:
        p = np.where(probs[:, 2] >= probs[:, 0], "positive", "negative"); labs = ["negative", "positive"]
    else:
        p = np.array(C3)[probs.argmax(1)]; labs = C3
    conf = {g: {q: int(((y == g) & (p == q)).sum()) for q in labs} for g in labs}   # gold -> predicted counts
    return dict(n=len(df), acc=float((p == y).mean()), macro_f1=macro_f1(y, p, labs), conf=conf)


def summarize(r):
    out = {}
    for fam in ["tweets", "amazon", "mteb"]:
        v = [x["macro_f1"] for k, x in r.items() if k.startswith(fam + "_")]
        if v:
            out[f"_{fam}_mean_f1"] = float(np.mean(v))
    return out


def hard_exit(code=0):
    """os._exit after killing our child processes: a vLLM engine core left alive keeps the job's log pipe open and the job
    never ends (2026-09-30: an eval job ran on for 30 min after writing its results)."""
    import signal, sys
    sys.stdout.flush(); me = str(os.getpid())
    for p in os.listdir("/proc"):
        try:
            if p.isdigit() and open(f"/proc/{p}/stat").read().rsplit(")", 1)[1].split()[1] == me:
                os.kill(int(p), signal.SIGKILL)
        except Exception:
            pass
    os._exit(code)
