"""Evaluate irony/sarcasm classifiers on irony/evals.   python irony/eval_irony.py --models SPEC,SPEC --evals irony/evals --out R.json
SPEC = model[::label] picks the output used as the irony score (default: the label named irony/ironic/sarcasm/sarcastic/LABEL_1, or
the non-negative class of a binary model). Models get the reply text only, except ours (config 'uses_context': true) which also get
the post as a text pair. ROC AUC and F1 at 0.5 per set; _mpico_mean_auc (9 languages), _tweets_mean_auc (semeval18, isarcasm,
ar_sarcasm, ko_sarcasm)."""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import auc, f1

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()
POS = {"irony", "ironic", "sarcasm", "sarcastic", "label_1"}
NEG = {"non_irony", "not_irony", "non-irony", "not_sarcastic", "non_sarcastic", "label_0", "not", "no", "regular"}
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
    multi = m.config.problem_type == "multi_label_classification"; ctx = bool(getattr(m.config, "uses_context", False))
    col = labels.index(lab) if lab else next((i for i, l in enumerate(low) if l in POS), None)
    if col is None and len(labels) == 2:
        col = next(i for i, l in enumerate(low) if l not in NEG)
    print(spec, "labels", labels, "col", col, "context", ctx, flush=True)

    @torch.no_grad()
    def run(enc_list):
        out = np.zeros(len(enc_list)); order = np.argsort([len(e) for e in enc_list])
        for s in range(0, len(enc_list), args.bs):
            b = order[s:s + args.bs]; L = max(len(enc_list[i]) for i in b)
            ids = torch.full((len(b), L), tok.pad_token_id); att = torch.zeros((len(b), L), dtype=torch.long)
            for j, i in enumerate(b):
                ids[j, :len(enc_list[i])] = torch.tensor(enc_list[i]); att[j, :len(enc_list[i])] = 1
            lo = m(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()
            out[b] = (torch.sigmoid(lo) if multi else torch.softmax(lo, -1))[:, col].cpu().numpy()
        return out

    def scores(texts, contexts):  # same encoding as tox/train_tox.py --context: pairs only where a context is given
        enc = [None] * len(texts); single = [i for i, c in enumerate(contexts) if not (ctx and c)]; s1 = set(single); pair = [i for i in range(len(texts)) if i not in s1]
        for i, e in zip(single, tok([texts[i] for i in single], truncation=True, max_length=args.max_len)["input_ids"] if single else []):
            enc[i] = e
        for i, e in zip(pair, tok([contexts[i] for i in pair], [texts[i] for i in pair], truncation="longest_first", max_length=args.max_len)["input_ids"] if pair else []):
            enc[i] = e
        return run(enc)
    r = {"labels": labels, "col": labels[col], "context": ctx}
    for name, df in EV.items():
        y = df.label.values; s = scores(df.text.tolist(), df.context.fillna("").tolist())
        r[name] = dict(n=len(df), auc=auc(y, s), f1_05=f1(y, s >= 0.5))
        if ctx and (df.context.fillna("") != "").any():
            s0 = scores(df.text.tolist(), [""] * len(df)); r[name].update(auc_noctx=auc(y, s0))
    mp = [k for k in r if k.startswith("mpico_")]
    r["_mpico_mean_auc"] = float(np.mean([r[k]["auc"] for k in mp]))
    r["_tweets_mean_auc"] = float(np.mean([r[k]["auc"] for k in ("semeval18", "isarcasm", "ar_sarcasm", "ko_sarcasm") if k in r]))
    print(spec, {k: round(v, 4) for k, v in r.items() if k.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
