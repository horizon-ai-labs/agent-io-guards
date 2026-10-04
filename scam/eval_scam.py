"""Evaluate classifiers on scam/evals.   python scam/eval_scam.py OUT.json EVAL_DIR MODEL[@i+j] [...]
@i+j: the class indices whose probabilities sum to the "unsafe" score (default: every class but 0)."""
import json, os, sys
import numpy as np, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_evals, score, summarize

out, ev = sys.argv[1:3]; EV = load_evals(ev)
res = json.load(open(out)) if os.path.exists(out) else {}
for spec in sys.argv[3:]:
    name, _, idx = spec.partition("@")
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
    n = m.config.num_labels; un = [int(i) for i in idx.split("+")] if idx else list(range(1, n))
    print(name, "labels", m.config.id2label, "unsafe", un, flush=True)
    r = {}
    for sname, df in EV.items():
        texts = df.text.tolist(); order = np.argsort([len(t) for t in texts]); P = np.zeros(len(texts))
        with torch.no_grad():
            for s in range(0, len(texts), 32):
                b = order[s:s + 32]
                enc = tok([texts[i] for i in b], truncation=True, max_length=512, padding=True, return_tensors="pt")
                lo = m(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.float()
                P[b] = torch.softmax(lo, -1)[:, un].sum(-1).cpu().numpy() if n > 1 else torch.sigmoid(lo[:, 0]).cpu().numpy()
        r[sname] = score(df, P)
    r.update(summarize(r)); print(name, {k: round(v, 4) for k, v in r.items() if k.startswith("_")}, flush=True)
    res[name] = r; json.dump(res, open(out, "w"), indent=1)
