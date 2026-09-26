"""Evaluate content-safety classifiers on safety/evals/*.parquet (+ optional in-domain test parquet).

python safety/eval_safety.py --models M1,M2 --evals DIR --out results.json
Model specs:
  <path or hub id>              sequence classifier; unsafe score = the "unsafe" / "toxicity" / "toxic" label (else label 1),
                                sigmoid for multi-label or 1-logit heads, softmax otherwise. Prompt-response items are fed
                                as a text pair only to models that were trained on pairs (ours, PAIR_MODELS); others get
                                the response alone.
  qwen3guard:<hub id>           Qwen3Guard-Gen via safety/qwenguard.py (prefilled "Safety:", next-token probabilities).
                                strict = unsafe + controversial, loose = unsafe only (both reported).
Metrics at threshold 0.5: f1, precision, recall, fpr, auc (when both classes occur), per-language f1 / recall.
"""
import argparse, glob, json, os, sys, time
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, roc_auc_score, precision_score

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--evals", required=True)
ap.add_argument("--extra", default="", help="comma list of extra parquet files (name = file stem), e.g. in-domain test")
ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64)
args = ap.parse_args()
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

evals = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}
for p in [x for x in args.extra.split(",") if x]:
    d = pd.read_parquet(p)
    if "unsafe" in d.columns:
        d = d.rename(columns={"unsafe": "label"})
    evals["indomain_" + os.path.basename(p)[:-8]] = d[["text", "text_pair", "label", "lang"] + (["cats"] if "cats" in d else [])].sample(n=min(6000, len(d)), random_state=0)
PAIR_MODELS = ("/", "Horizon-Labs/")   # local dirs (our runs) and our Hub repos use (prompt, response) pairs


def metrics(y, s, lang):
    pred = s > 0.5
    r = dict(n=len(y), f1=float(f1_score(y, pred, zero_division=0)), precision=float(precision_score(y, pred, zero_division=0)),
             recall=float(pred[y == 1].mean()) if (y == 1).any() else None, fpr=float(pred[y == 0].mean()) if (y == 0).any() else None)
    r["auc"] = float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else None
    r["per_lang"] = {l: dict(f1=float(f1_score(y[m], pred[m], zero_division=0)), recall=float(pred[m][y[m] == 1].mean()) if (y[m] == 1).any() else None)
                     for l in sorted(set(lang)) for m in [lang == l]}
    return r


def cat_metrics(df, full, names):
    """Per-category quality on unsafe items of a set with gold category lists: AUC, F1@0.5, best-threshold F1."""
    m = df.label.values == 1; r = {}
    for k, c in enumerate(names):
        if c == "unsafe":
            continue
        y = df.cats.values[m]; y = np.array([c in list(x) for x in y]).astype(int); s = full[m, k]
        if y.sum() < 10:
            continue
        best = max((f1_score(y, s > t, zero_division=0), t) for t in np.arange(0.05, 0.95, 0.05))
        r[c] = dict(n_pos=int(y.sum()), auc=float(roc_auc_score(y, s)), f1_05=float(f1_score(y, s > 0.5, zero_division=0)),
                    f1_best=float(best[0]), thr_best=float(best[1]))
    return r


def classifier(spec):
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec)
    model = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    names = {i: l.lower() for i, l in model.config.id2label.items()}
    idx = next((i for key in ["unsafe", "toxicity", "toxic"] for i, l in names.items() if l == key), None)
    if idx is None:
        idx = 1 if len(names) > 1 else 0
    sig = model.config.problem_type == "multi_label_classification" or len(names) == 1
    print(f"[{spec}] labels {list(names.values())[:20]} -> index {idx} ({names[idx]}), {'sigmoid' if sig else 'softmax'}", flush=True)
    pair = spec.startswith(PAIR_MODELS)
    maxlen = min(1024 if pair else 512, getattr(tok, "model_max_length", 512) or 512)

    @torch.no_grad()
    def score(df):
        out, full = np.zeros(len(df)), np.zeros((len(df), len(names)))
        order = np.argsort([len(a) + len(b) for a, b in zip(df.text, df.text_pair)])
        for s in range(0, len(df), args.bs):
            b = order[s: s + args.bs]
            t, p = df.text.values[b].tolist(), df.text_pair.values[b].tolist()
            if pair and any(p):
                enc = tok(t, p, truncation="longest_first", max_length=maxlen, padding=True, return_tensors="pt")
            else:
                enc = tok([q if q else x for x, q in zip(t, p)], truncation=True, max_length=maxlen, padding=True, return_tensors="pt")
            lo = model(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.float()
            pr = torch.sigmoid(lo) if sig else torch.softmax(lo, -1)
            out[b] = pr[:, idx].cpu().numpy()
            full[b] = pr.cpu().numpy()
        return {"": out, "_full": full}
    score.names = [names[i] for i in range(len(names))]
    return score


def qwen3guard(spec):
    from qwenguard import QwenGuard
    g = QwenGuard(spec); g.sanity()

    def score(df):
        pr = g.score(df.text.tolist(), df.text_pair.tolist())
        return {"_strict": pr[:, 1] + pr[:, 2], "_loose": pr[:, 1]}
    return score


res = {}
for spec in args.models.split(","):
    t0 = time.time()
    fn = qwen3guard(spec.split(":", 1)[1]) if spec.startswith("qwen3guard:") else classifier(spec)
    outs = {}
    for name, df in evals.items():
        sc = fn(df)
        full = sc.pop("_full", None)
        if full is not None and "cats" in df.columns and getattr(fn, "names", None) and full.shape[1] > 2:
            outs.setdefault("", {})["categories_" + name] = cat_metrics(df, full, fn.names)
        for suf, s in sc.items():
            outs.setdefault(suf, {})[name] = metrics(df.label.values, s, df.lang.values)
        print(spec, name, {suf: round(outs[suf][name]["f1"], 3) for suf in sc}, flush=True)
    for suf, r in outs.items():
        res[spec + suf] = dict(r, _meta=dict(seconds=time.time() - t0))
    json.dump(res, open(args.out, "w"), indent=1)
    del fn; torch.cuda.empty_cache()
print("done")
