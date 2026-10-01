"""Evaluate toxicity classifiers.   python tox/eval_tox.py --models A,B --evals tox/evals --out R.json
civil: ROC AUC per Detoxify label on Civil Comments test (labels binarised at >= 0.5; labels with < 20 positives skipped),
mean over labels the model has. tdx_<lang>: ROC AUC and F1 at 0.5 of the model's toxicity score on TextDetox.
Label mapping by name: toxicity|toxic|LABEL_1 (binary) -> toxicity; severe_toxic -> severe_toxicity; identity_hate ->
identity_attack; neutral / non-toxic / not_toxic = the complement. Multi-label models use sigmoid, single-label softmax.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()
L7 = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
ALIAS = {"toxic": "toxicity", "toxicity": "toxicity", "severe_toxic": "severe_toxicity", "severe_toxicity": "severe_toxicity", "obscene": "obscene",
         "threat": "threat", "insult": "insult", "identity_hate": "identity_attack", "identity_attack": "identity_attack",
         "sexual_explicit": "sexual_explicit"}
NEG = {"neutral", "non-toxic", "not_toxic", "nontoxic", "label_0"}


def auc(y, s):
    y, s = np.asarray(y).astype(bool), np.asarray(s, float)
    if y.all() or not y.any():
        return None
    order = s.argsort(); r = np.empty(len(s)); r[order] = np.arange(1, len(s) + 1)
    for v in np.unique(s[np.diff(np.sort(s), prepend=np.nan) == 0]):
        m = s == v; r[m] = r[m].mean()
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def f1(y, p):
    tp = (y & p).sum(); return 0.0 if tp == 0 else float(2 * tp / (2 * tp + (~y & p).sum() + (y & ~p).sum()))


EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec)
    m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    labels = [m.config.id2label[i].lower() for i in range(m.config.num_labels)]
    multi = m.config.problem_type == "multi_label_classification" or len(labels) == 1 or (len(labels) > 2 and not any(l in NEG for l in labels))
    col = {}
    for i, l in enumerate(labels):
        if l in ALIAS:
            col.setdefault(ALIAS[l], i)
    if "toxicity" not in col and len(labels) == 2:
        col["toxicity"] = next(i for i, l in enumerate(labels) if l not in NEG)
    print(spec, "labels", labels, "multi", multi, "mapped", col, flush=True)

    @torch.no_grad()
    def scores(texts):
        out = np.zeros((len(texts), len(labels))); order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), args.bs):
            b = order[s:s + args.bs]
            e = tok([texts[i] for i in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            lo = m(**{k: v.cuda() for k, v in e.items() if k in ("input_ids", "attention_mask")}).logits.float()
            out[b] = (torch.sigmoid(lo) if multi else torch.softmax(lo, -1)).cpu().numpy()
        return out
    r = {"labels": labels, "mapped": list(col)}
    for name, df in EV.items():
        S = scores(df.text.tolist())
        if name == "civil_test":
            per = {}
            for l in L7:
                y = df[l].values >= 0.5
                if l in col and y.sum() >= 20:
                    per[l] = auc(y, S[:, col[l]])
            r[name] = dict(n=len(df), auc=per, mean_auc=float(np.mean(list(per.values()))), toxicity_auc=per.get("toxicity"))
        else:
            y = df.toxic.values.astype(bool); s = S[:, col["toxicity"]]
            r[name] = dict(n=len(df), auc=auc(y, s), f1_05=f1(y, s >= 0.5))
    tdx = [v["auc"] for k, v in r.items() if k.startswith("tdx_") and v["auc"] is not None]
    r["_civil_mean_auc"] = r["civil_test"]["mean_auc"]; r["_civil_toxicity_auc"] = r["civil_test"]["toxicity_auc"]; r["_tdx_mean_auc"] = float(np.mean(tdx))
    r["_tdx_mean_f1"] = float(np.mean([v["f1_05"] for k, v in r.items() if k.startswith("tdx_")]))
    print(spec, {k: round(v, 4) for k, v in r.items() if k.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
