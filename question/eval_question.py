"""Evaluate question-vs-statement classifiers (2-class softmax, LABEL_1 = question) on question/evals:
sentences (held-out web sentences, 66 languages, labels from question marks, punctuation removed) and queries (held-out search queries,
47 languages, question vs keywords). Metrics: accuracy, macro-F1; sentences also per-language balanced accuracy (languages with >= 20
questions; Thai, Burmese, Khmer rarely use question marks) and English accuracy.   python question/eval_question.py --models A,B --out R.json"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__))
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
EV = {n: pd.read_parquet(os.path.join(HERE, "evals", f"{n}.parquet")).reset_index(drop=True) for n in ("sentences", "queries")}


def f1(y, p):
    out = []
    for c in (0, 1):
        tp = ((p == c) & (y == c)).sum(); out.append(2 * tp / max(1, 2 * tp + ((p == c) & (y != c)).sum() + ((p != c) & (y == c)).sum()))
    return float(np.mean(out))


res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec); m = AutoModelForSequenceClassification.from_pretrained(spec).cuda().eval()
    r = {}
    for name, df in EV.items():
        P = np.zeros(len(df)); order = np.argsort(df.text.str.len().values)
        with torch.no_grad():
            for s in range(0, len(df), 128):
                b = order[s:s + 128]; e = tok(df.text.iloc[b].tolist(), truncation=True, max_length=128, padding=True, return_tensors="pt")
                P[b] = torch.softmax(m(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).logits.float(), -1)[:, 1].cpu().numpy()
        y = df.label.values; p = (P >= 0.5).astype(int); x = dict(n=len(df), acc=float((p == y).mean()), macro_f1=f1(y, p))
        bal = {}
        for l, g in df.groupby("lang"):
            if (g.label == 1).sum() >= 20 and (g.label == 0).sum() >= 20:
                yy, pp = g.label.values, p[g.index.values]; bal[l] = float(((pp[yy == 1] == 1).mean() + (pp[yy == 0] == 0).mean()) / 2)
        x["lang_bal_acc"] = bal; x["lang_mean_bal_acc"] = float(np.mean(list(bal.values())))
        en = df.lang.isin(["eng_Latn"]).values; x["acc_en"] = float((p[en] == y[en]).mean()) if en.any() else None
        r[name] = x; print(spec, name, {k: round(v, 4) for k, v in x.items() if isinstance(v, float)}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
