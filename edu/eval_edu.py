"""Evaluate educational-quality regressors (FineWeb-Edu style: one regression output, 0-5) on edu/evals.
python edu/eval_edu.py --models A,B --evals edu/evals --out R.json
Per set (reference column 'score'): Spearman and Pearson correlation, F1 of 'educational' (reference >= 3 vs prediction >= 2.5, the
FineWeb-Edu int_score >= 3 cut), accuracy of the rounded prediction."""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import f1

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=32); ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()
EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    try:
        tok = AutoTokenizer.from_pretrained(spec)
        m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    except Exception as e:
        print(spec, "LOAD FAILED", repr(e)[:300], flush=True); continue
    ml = min(args.max_len, getattr(m.config, "max_position_embeddings", 512) - 2 if getattr(m.config, "model_type", "") in ("xlm-roberta", "roberta") else args.max_len)

    @torch.no_grad()
    def scores(texts):
        out = np.zeros(len(texts)); order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), args.bs):
            b = order[s:s + args.bs]
            e = tok([texts[i] for i in b], truncation=True, max_length=ml, padding=True, return_tensors="pt")
            out[b] = m(**{k: v.cuda() for k, v in e.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits.float()[:, 0].cpu().numpy()
        return out
    r = {}
    for name, df in EV.items():
        s = scores(df.text.tolist()); y = df.score.values
        r[name] = dict(n=len(df), spearman=float(np.corrcoef(pd.Series(s).rank(), pd.Series(y).rank())[0, 1]), pearson=float(np.corrcoef(s, y)[0, 1]),
                       f1_ge3=f1(y >= 2.5, s >= 2.5), acc_round=float((np.clip(np.round(s), 0, 5) == np.round(y)).mean()), mean_pred=float(s.mean()))
        if "lang" in df:
            rk = lambda a: pd.Series(a).rank().values
            r[name]["by_lang"] = {l: float(np.corrcoef(rk(s[g.index]), rk(y[g.index]))[0, 1]) for l, g in df.groupby("lang")}
            r[name]["lang_mean_spearman"] = float(np.mean(list(r[name]["by_lang"].values())))
        print(spec, name, {k: round(v, 4) for k, v in r[name].items() if not isinstance(v, dict)}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
