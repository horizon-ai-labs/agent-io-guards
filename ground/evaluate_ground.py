"""Groundedness benchmark: does the response follow from the context?  Label 1 = supported, 0 = unsupported.

python ground/evaluate_ground.py --models a,b --out results.json [--ours_data <data dir with eval/*.parquet>]
Metrics: balanced accuracy at 0.5 (the LLM-AggreFact convention) and ROC AUC.
Models: our sequence-pair classifiers, MiniCheck RoBERTa/DeBERTa (pair classifiers, label 1 = supported),
vectara/hallucination_evaluation_model (HHEM-2.1-open, trust_remote_code, .predict(pairs) -> consistency score).
"""
import argparse, glob, json, os, random, time
import numpy as np, torch
from datasets import load_dataset
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ours_data", default="")
ap.add_argument("--max_n", type=int, default=2000)
ap.add_argument("--max_len", type=int, default=2048)
ap.add_argument("--extra_evals", default="", help="comma list of dirs with *.parquet (text, text_pair, label)")
args = ap.parse_args()
rng = random.Random(0)


def bench():
    B = {}
    # RAGTruth test: response-level (unsupported if any hallucination span)
    rows = []
    for r in load_dataset("wandb/RAGTruth-processed", split="test"):
        labs = r["hallucination_labels_processed"]
        hal = bool(labs) and any(v for v in (labs.values() if isinstance(labs, dict) else [labs]))
        ctx = (r["query"] or "") + "\n\n" + (r["context"] or "")
        rows.append((ctx.strip(), r["output"], 0 if hal else 1))
    B["ragtruth_test"] = rows
    # HaluEval (Apache-2.0): each item gives a right and a hallucinated response
    for cfg, kf, qf, rf, hf in [("qa", "knowledge", "question", "right_answer", "hallucinated_answer"),
                                ("dialogue", "knowledge", "dialogue_history", "right_response", "hallucinated_response"),
                                ("summarization", "document", None, "right_summary", "hallucinated_summary")]:
        ds = list(load_dataset("pminervini/HaluEval", cfg, split="data"))
        rng.shuffle(ds)
        rows = []
        for r in ds[: args.max_n // 2]:
            ctx = r[kf] + (("\n\n" + r[qf]) if qf else "")
            rows += [(ctx, r[rf], 1), (ctx, r[hf], 0)]
        B[f"halueval_{cfg}"] = rows
    if args.ours_data:
        import pandas as pd
        for p in sorted(glob.glob(f"{args.ours_data}/eval/*.parquet")):
            df = pd.read_parquet(p).sample(frac=1.0, random_state=0).head(args.max_n)
            B["ours_" + os.path.basename(p)[:-8]] = list(zip(df.text, df.text_pair, df.label))
    for d in [x for x in args.extra_evals.split(",") if x]:
        import pandas as pd
        for p in sorted(glob.glob(f"{d}/*.parquet")):
            df = pd.read_parquet(p)
            B[os.path.basename(p)[:-8]] = list(zip(df.text, df.text_pair, df.label))
    for k in B:
        B[k] = B[k][: args.max_n]
        print("bench", k, len(B[k]), "pos", sum(x[2] for x in B[k]), flush=True)
    return B


def pos_idx(model):
    lab = {int(k): v.upper() for k, v in model.config.id2label.items()}
    for i, v in lab.items():
        if v in ("SUPPORTED", "CONSISTENT", "ENTAILMENT", "SUPPORT"):
            return i
    return 1


@torch.no_grad()
def scorer(name):
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    if "hallucination_evaluation_model" in name:
        from transformers import AutoModelForSequenceClassification as A
        m = A.from_pretrained(name, trust_remote_code=True).cuda().eval()
        # HHEM has no input limit and OOMs on very long documents: cap contexts at 12k characters (noted in results)
        return lambda pairs: np.array([float(x) for b in range(0, len(pairs), 2)
                                       for x in m.predict([(c[:12000], r) for c, r in pairs[b:b + 2]])])
    tok = AutoTokenizer.from_pretrained(name)
    m = AutoModelForSequenceClassification.from_pretrained(name).cuda().eval()
    pi = pos_idx(m)
    ml = min(args.max_len, tok.model_max_length if tok.model_max_length < 100000 else 512)

    def chunks(pairs):
        """Short-window models (e.g. MiniCheck, 512): split the context into chunks that fit next to the response
        and take the max support over chunks, as MiniCheck's own library does. 8k-context models get one chunk."""
        items, owner = [], []
        for i, (c, r) in enumerate(pairs):
            rl = len(tok(r, add_special_tokens=False).input_ids)
            room = max(64, ml - rl - 8)
            ids = tok(c, add_special_tokens=False).input_ids
            if len(ids) <= room:
                items.append((c, r)); owner.append(i); continue
            step = max(32, room - room // 8)
            for s in range(0, len(ids), step):
                items.append((tok.decode(ids[s:s + room]), r)); owner.append(i)
                if s + room >= len(ids):
                    break
        return items, owner

    def f(pairs):
        items, owner = chunks(pairs)
        sc = score_items(items)
        out = np.zeros(len(pairs))
        for o, v in zip(owner, sc):
            out[o] = max(out[o], v)
        return out

    def score_items(pairs):
        out = []
        for b in range(0, len(pairs), 16):
            c = [p[0] for p in pairs[b:b + 16]]; r = [p[1] for p in pairs[b:b + 16]]
            enc = tok(c, r, truncation="longest_first", max_length=ml, padding=True, return_tensors="pt").to("cuda")
            enc.pop("token_type_ids", None) if "modernbert" in m.config.model_type else None
            with torch.autocast("cuda", dtype=torch.bfloat16):
                lo = m(**enc).logits.float()
            out += torch.softmax(lo, -1)[:, pi].cpu().tolist()
        return np.array(out)
    return f


B = bench()
results = json.load(open(args.out)) if os.path.exists(args.out) else {}
for name in args.models.split(","):
    t0 = time.time()
    try:
        f = scorer(name)
    except Exception as e:
        print("LOAD FAIL", name, repr(e)[:300], flush=True); results[name] = {"error": repr(e)[:300]}; continue
    res = {}
    for b, rows in B.items():
        y = np.array([r[2] for r in rows]); p = f([(r[0], r[1]) for r in rows])
        res[b] = dict(bacc=float(balanced_accuracy_score(y, p > 0.5)), auc=float(roc_auc_score(y, p)) if 0 < y.sum() < len(y) else None, n=len(y))
        print(f"{name[-50:]:50s} {b:24s} bacc={res[b]['bacc']:.3f} auc={res[b]['auc'] or 0:.3f}", flush=True)
    res["_meta"] = dict(seconds=time.time() - t0)
    results[name] = res
    json.dump(results, open(args.out, "w"), indent=1)
    torch.cuda.empty_cache()
