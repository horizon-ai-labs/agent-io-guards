"""Evaluate hate-speech classifiers on hate/evals.   python hate/eval_hate.py --models SPEC,SPEC --evals hate/evals --out R.json
SPEC = model[::label] picks the output used as the hate score (default: the label named hate/hateful/HATE/HATEFUL, or the
non-negative class of a binary model); model::!label uses 1 - P(label). Multi-label models (problem_type) use sigmoid, others softmax.
Per set: ROC AUC and F1 at 0.5 of the hate score; MHC also per functionality (accuracy at 0.5); MHS also target-group AUC on its
hateful comments when the model has target_<group> outputs. Summary keys: _mhc_mean_auc (11 languages), _mhc_mean_f1, _real_mean_auc
(mhs, hatexplain, kmhas), _heldout_real_mean_auc (hs18, ethos, olidbr, kmhas; the v2 gate)."""
import argparse, glob, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import auc, f1

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()
POS = {"hate", "hateful", "hatespeech", "hate_speech"}
NEG = {"nothate", "not-hate", "non_hate", "not", "label_0", "neutral", "normal", "not_hate"}
HELD = ("hs18", "ethos", "olidbr", "kmhas")  # v2 gate sets (hate/build_evals2.py + K-MHaS)
G = ["race", "religion", "origin", "gender", "sexuality", "age", "disability"]


EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    path, _, lab = spec.partition("::")
    try:
        tok = AutoTokenizer.from_pretrained(path)
        m = AutoModelForSequenceClassification.from_pretrained(path, torch_dtype=torch.bfloat16).cuda().eval()
    except Exception as e:
        print(spec, "LOAD FAILED", repr(e)[:300], flush=True); continue
    labels = [m.config.id2label[i] for i in range(m.config.num_labels)]; low = [l.lower() for l in labels]
    multi = m.config.problem_type == "multi_label_classification"
    neg = lab.startswith("!"); lab = lab.lstrip("!")
    if lab:
        col = labels.index(lab)
    else:
        col = next((i for i, l in enumerate(low) if l in POS), None)
        if col is None and len(labels) == 2:
            col = next(i for i, l in enumerate(low) if l not in NEG)
    tcols = {g: labels.index(f"target_{g}") for g in G if f"target_{g}" in labels}
    print(spec, "labels", labels, "multi", multi, "col", col, "neg", neg, flush=True)

    @torch.no_grad()
    def scores(texts):
        out = np.zeros((len(texts), len(labels))); order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), args.bs):
            b = order[s:s + args.bs]
            e = tok([texts[i] for i in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            lo = m(**{k: v.cuda() for k, v in e.items() if k in ("input_ids", "attention_mask")}).logits.float()
            out[b] = (torch.sigmoid(lo) if multi else torch.softmax(lo, -1)).cpu().numpy()
        return out
    r = {"labels": labels, "col": labels[col], "neg": neg}
    for name, df in EV.items():
        S = scores(df.text.tolist()); s = 1 - S[:, col] if neg else S[:, col]; y = df.hate.values
        r[name] = dict(n=len(df), auc=auc(y, s), f1_05=f1(y, s >= 0.5), fpr_05=float((s[y == 0] >= 0.5).mean()))
        if name.startswith("mhc_"):
            r[name]["func_acc"] = {fn: float(((s[g.index] >= 0.5) == g.hate.values.astype(bool)).mean()) for fn, g in df.groupby("functionality")}
        if name == "mhs" and tcols:
            h = df.hate.values == 1
            r[name]["target_auc"] = {g: auc(df[f"target_{g}"].values[h], S[h, c]) for g, c in tcols.items()}
        print(spec, name, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r[name].items() if k in ("auc", "f1_05", "fpr_05")}, flush=True)
    mhc = [k for k in r if k.startswith("mhc_")]
    if mhc:
        r["_mhc_mean_auc"] = float(np.mean([r[k]["auc"] for k in mhc])); r["_mhc_mean_f1"] = float(np.mean([r[k]["f1_05"] for k in mhc]))
    if all(k in r for k in ("mhs", "hatexplain", "kmhas")):
        r["_real_mean_auc"] = float(np.mean([r[k]["auc"] for k in ("mhs", "hatexplain", "kmhas")]))
    if all(k in r for k in HELD):
        r["_heldout_real_mean_auc"] = float(np.mean([r[k]["auc"] for k in HELD]))
    print(spec, {k: round(v, 4) for k, v in r.items() if k.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
