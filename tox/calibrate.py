"""Map teacher P(yes) per label onto the Civil Comments label scale (fraction of annotators), by isotonic regression (PAV) fit on
v1.0 English training comments scored by the teacher (tox/run_label.sh LABEL_IN=.gated/tox/calib_in.parquet LABEL_NAME=calib).
The raw teacher is far more confident (e.g. severe_toxicity .72 vs .03 on toxic comments), which would distort the student's heads.
python tox/calibrate.py OUT_DIR CALIB_SHARDS... -- NATIVE_SHARDS...   -> OUT_DIR/native_cal.parquet, OUT_DIR/calibration.json"""
import json, os, sys
import numpy as np, pandas as pd
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
out = sys.argv[1]; i = sys.argv.index("--")
cal = pd.concat([pd.read_parquet(p) for p in sys.argv[2:i]], ignore_index=True)
nat = pd.concat([pd.read_parquet(p) for p in sys.argv[i + 1:]], ignore_index=True)
ref = pd.read_parquet(".gated/tox/calib_in.parquet").set_index("text")
y_all = ref.loc[cal.text, LABELS].values  # true Civil labels; cal holds the teacher scores in the label columns


def pav(x, y):
    o = np.argsort(x, kind="stable"); xs, ys = x[o], y[o]
    blocks = []  # [sum, count, x_max]
    for a, b in zip(xs, ys):
        blocks.append([b, 1, a])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, m = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += c; blocks[-1][2] = m
    return np.array([b[2] for b in blocks]), np.array([b[0] / b[1] for b in blocks])


def auc(y, s):
    y = np.asarray(y) >= 0.5; r = pd.Series(s).rank().values
    return float((r[y].sum() - y.sum() * (y.sum() + 1) / 2) / (y.sum() * (~y).sum())) if 0 < y.sum() < len(y) else None


rep = {}
for j, l in enumerate(LABELS):
    x, y = cal[l].values, y_all[:, j]
    xm, ym = pav(x, y)
    f = lambda s: ym[np.minimum(np.searchsorted(xm, s, side="left"), len(xm) - 1)]
    raw = nat[l].values; nat[l] = f(raw)
    rep[l] = dict(teacher_auc_on_civil=auc(y, x), civil_mean=round(float(y.mean()), 4), teacher_mean=round(float(x.mean()), 4),
                  native_mean_raw=round(float(raw.mean()), 4), native_mean_cal=round(float(nat[l].mean()), 4),
                  map={str(q): round(float(f(np.array([q]))[0]), 3) for q in (0.05, 0.25, 0.5, 0.75, 0.95, 0.99)})
    print(l, json.dumps(rep[l]), flush=True)
os.makedirs(out, exist_ok=True); nat.to_parquet(f"{out}/native_cal.parquet"); json.dump(rep, open(f"{out}/calibration.json", "w"), indent=1)
