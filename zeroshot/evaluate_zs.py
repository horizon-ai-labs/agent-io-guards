"""Zero-shot classification benchmark for NLI models.  python zeroshot/evaluate_zs.py --models a,b --evals DIR --out res.json

Classification: each (text, template.format(label)) pair is scored and the label with the highest entailment logit
wins, as in transformers' zero-shot-classification pipeline (multi_label=False). Metrics: accuracy and macro-F1,
per language and overall. XNLI: entailment vs not, P(entailment) > 0.5 over all classes, balanced accuracy.
"""
import argparse, glob, json, os, time
from collections import defaultdict
import numpy as np, torch
from sklearn.metrics import balanced_accuracy_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--evals", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--bs", type=int, default=128)
ap.add_argument("--only", default="", help="comma list of eval names")
args = ap.parse_args()


def ent_idx(cfg):
    for k, v in cfg.id2label.items():
        if v.lower().startswith("entail"):
            return int(k)
    raise ValueError(f"no entailment label in {cfg.id2label}")


def load(name):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, torch_dtype=torch.bfloat16).cuda().eval()
    ei = ent_idx(m.config)

    @torch.no_grad()
    def score(pairs):
        """-> (entailment logit, P(entailment)) per pair; length-sorted batches."""
        order = sorted(range(len(pairs)), key=lambda i: len(pairs[i][0]) + len(pairs[i][1]))
        lo, pr = np.zeros(len(pairs)), np.zeros(len(pairs))
        for b in range(0, len(order), args.bs):
            idx = order[b:b + args.bs]
            enc = tok([pairs[i][0] for i in idx], [pairs[i][1] for i in idx], truncation="only_first",
                      max_length=args.max_len, padding=True, return_tensors="pt").to("cuda")
            if m.config.model_type in ("modernbert",):
                enc.pop("token_type_ids", None)
            out = m(**enc).logits.float()
            lo[idx] = out[:, ei].cpu().numpy(); pr[idx] = torch.softmax(out, -1)[:, ei].cpu().numpy()
        return lo, pr
    return score


def run_cls(score, d):
    labels, items = d["labels"], d["items"]
    hyps = [d["template"].format(l) for l in labels]
    pairs = [(it["text"], h) for it in items for h in hyps]
    lo, _ = score(pairs)
    pred = lo.reshape(len(items), len(labels)).argmax(1)
    gold = np.array([it["gold"] for it in items]); langs = np.array([it["lang"] for it in items])
    res = {}
    for l in ["all"] + sorted(set(langs)):
        mk = np.ones(len(items), bool) if l == "all" else langs == l
        res[l] = dict(acc=float((pred[mk] == gold[mk]).mean()), f1=float(f1_score(gold[mk], pred[mk], average="macro")), n=int(mk.sum()))
    return res, pred.tolist()


def run_nli(score, d):
    items = d["items"]
    _, pr = score([(it["premise"], it["hypothesis"]) for it in items])
    y = np.array([it["label"] for it in items]); langs = np.array([it["lang"] for it in items])
    res = {}
    for l in ["all"] + sorted(set(langs)):
        mk = np.ones(len(items), bool) if l == "all" else langs == l
        res[l] = dict(bacc=float(balanced_accuracy_score(y[mk], pr[mk] > 0.5)), n=int(mk.sum()))
    return res, pr.round(4).tolist()


E = {os.path.basename(p)[:-5]: json.load(open(p)) for p in sorted(glob.glob(f"{args.evals}/*.json"))}
if args.only:
    E = {k: v for k, v in E.items() if k in args.only.split(",")}
results = json.load(open(args.out)) if os.path.exists(args.out) else {}
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
for name in args.models.split(","):
    t0 = time.time()
    try:
        score = load(name)
    except Exception as e:
        print("LOAD FAIL", name, repr(e)[:300], flush=True); results[name] = {"error": repr(e)[:300]}; continue
    res, preds = results.get(name, {}), {}
    for k, d in E.items():
        t1 = time.time()
        r, p = (run_nli if d.get("type") == "nli" else run_cls)(score, d)
        res[k] = r; preds[k] = p
        main = r["all"].get("acc", r["all"].get("bacc"))
        print(f"{name[-45:]:45s} {k:10s} {main:.3f}  " + " ".join(f"{l}={v.get('acc', v.get('bacc')):.2f}" for l, v in r.items() if l != "all")
              + f"  ({time.time() - t1:.0f}s)", flush=True)
    res["_meta"] = dict(seconds=time.time() - t0)
    results[name] = res
    json.dump(results, open(args.out, "w"), indent=1)
    json.dump(preds, open(args.out[:-5] + "_preds_" + name.replace("/", "__") + ".json", "w"))
    torch.cuda.empty_cache()
