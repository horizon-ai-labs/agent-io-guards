"""Evaluate transformers sentiment classifiers on sentiment/evals.   python sentiment/eval_sent.py --models A,B --evals DIR --out R.json
Each model's labels are mapped to negative/neutral/positive by name (very negative -> negative; 1-2 stars -> negative,
3 stars -> neutral, 4-5 stars -> positive; LABEL_0/1/2 -> negative/neutral/positive) and probabilities summed per class.
"""
import argparse, json, os, re, sys
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_evals, score, summarize

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()


def cls_of(name, n):
    s = name.lower()
    if "neg" in s: return 0
    if "neu" in s: return 1
    if "pos" in s: return 2
    m = re.match(r"(\d) star", s)
    if m: return 0 if int(m.group(1)) <= 2 else 1 if int(m.group(1)) == 3 else 2
    m = re.match(r"label_(\d)", s)
    if m and n == 3: return int(m.group(1))
    raise ValueError(f"unmapped label {name}")


EV = load_evals(args.evals)
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec)
    m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    id2 = m.config.id2label; M = torch.zeros(len(id2), 3)
    for i, l in id2.items():
        M[int(i), cls_of(l, len(id2))] = 1
    print(spec, {l: cls_of(l, len(id2)) for l in id2.values()}, flush=True)
    M = M.cuda(); r = {}
    for name, df in EV.items():
        texts = df.text.tolist(); order = np.argsort([len(t) for t in texts]); P = np.zeros((len(texts), 3))
        with torch.no_grad():
            for s in range(0, len(texts), args.bs):
                b = order[s:s + args.bs]
                enc = tok([texts[i] for i in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
                lo = m(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.float()
                P[b] = (torch.softmax(lo, -1) @ M).cpu().numpy()
        r[name] = score(df, P)
    r.update(summarize(r)); print(spec, {k: round(v, 4) for k, v in r.items() if k.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
