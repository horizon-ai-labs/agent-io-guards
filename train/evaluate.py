"""Evaluate one or more classifiers on the eval suite.

python train/evaluate.py --evals $DATA/eval --models path_or_id[,...] --out results.json [--window 512]
Long inputs are split into overlapping windows; the document score is the max over windows.
"""
import argparse, glob, json, os, re, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score

ap = argparse.ArgumentParser()
ap.add_argument("--evals", required=True)
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--window", type=int, default=0, help="0 = min(model max, 2048)")
ap.add_argument("--normalize", action="store_true", help="prepend NFKC + invisible-char stripping (train/normalizer.py)")
ap.add_argument("--batch_tokens", type=int, default=131072)
args = ap.parse_args()

POS = re.compile(r"inject|jailbreak|malicious|unsafe|attack", re.I)
sets = {os.path.basename(p)[:-8]: pd.read_parquet(p) for d in args.evals.split(",") for p in sorted(glob.glob(f"{d}/*.parquet"))}
results = json.load(open(args.out)) if os.path.exists(args.out) else {}


def pos_index(model):
    labels = {int(k): v for k, v in model.config.id2label.items()}
    hits = [i for i, v in labels.items() if POS.search(v)]
    return hits[0] if hits else max(labels)  # LABEL_1 convention


@torch.no_grad()
def score(model, tok, texts, window):
    stride = window // 4
    enc = tok(texts, truncation=True, max_length=window, stride=stride, return_overflowing_tokens=True)
    mapping = enc["overflow_to_sample_mapping"]
    ids = enc["input_ids"]
    order = np.argsort([len(x) for x in ids])
    pi = pos_index(model)
    win_scores = np.zeros(len(ids))
    i = 0
    while i < len(order):
        L = len(ids[order[min(i + 1, len(order) - 1)]])
        bs = max(1, args.batch_tokens // max(1, len(ids[order[min(len(order) - 1, i + 255)]])))
        b = order[i : i + min(bs, 256)]
        L = max(len(ids[j]) for j in b)
        x = torch.full((len(b), L), tok.pad_token_id or 0, dtype=torch.long)
        a = torch.zeros((len(b), L), dtype=torch.long)
        for r, j in enumerate(b):
            x[r, : len(ids[j])] = torch.tensor(ids[j]); a[r, : len(ids[j])] = 1
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(input_ids=x.cuda(), attention_mask=a.cuda()).logits.float()
        win_scores[b] = torch.softmax(lo, -1)[:, pi].cpu().numpy()
        i += len(b)
    doc = np.zeros(len(texts))
    for w, d in enumerate(mapping):
        doc[d] = max(doc[d], win_scores[w])
    return doc


for name in args.models.split(","):
    t0 = time.time()
    tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
    if args.normalize:
        import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from normalizer import add_normalizer
        add_normalizer(tok)
    model = AutoModelForSequenceClassification.from_pretrained(name, trust_remote_code=True).cuda().eval()
    mx = getattr(model.config, "max_position_embeddings", 512)
    window = args.window or min(2048, mx if mx < 100000 else 512)
    if "deberta" in model.config.model_type:
        window = min(window, 512)
    res = {}
    for sname, df in sets.items():
        p = score(model, tok, df.text.tolist(), window)
        y = df.label.values
        pred = p > 0.5
        r = dict(n=len(y), pos=int(y.sum()), acc=float((pred == y).mean()))
        if 0 < y.sum() < len(y):
            r.update(auc=float(roc_auc_score(y, p)), f1=float(f1_score(y, pred)),
                     precision=float(precision_score(y, pred)), recall=float(recall_score(y, pred)),
                     fpr=float(pred[y == 0].mean()))
        res[sname] = r
        pdir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "preds", name.strip("/").replace("/", "__")[-80:])
        os.makedirs(pdir, exist_ok=True)
        df.assign(score=p).to_parquet(f"{pdir}/{sname}.parquet")
        print(f"{name:60s} {sname:24s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}={v}" for k, v in r.items()), flush=True)
    res["_meta"] = dict(window=window, seconds=time.time() - t0)
    results[name] = res
    json.dump(results, open(args.out, "w"), indent=1)
    del model; torch.cuda.empty_cache()
