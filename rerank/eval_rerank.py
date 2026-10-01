"""Evaluate rerankers on rerank/evals.   python rerank/eval_rerank.py --models A,B --evals rerank/evals --out R.json
Model specs: <hub id or dir> = cross-encoder (AutoModelForSequenceClassification; 1 logit, or logit[1] - logit[0] with 2);
qwen3rr:<hub id> = Qwen3-Reranker (causal LM; P(yes) with the model card's template and default web-search instruction).
Metric: nDCG@10 per set (graded relevance, all candidates of a query reranked), MRR@10; family means (_miracl, _wiki,
_other, _all). Query+document truncated to --max_len tokens (document side).
"""
import argparse, glob, json, math, os, time
import numpy as np, pandas as pd, torch

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--only", default="")
args = ap.parse_args()
EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))
      if not args.only or os.path.basename(p).startswith(tuple(args.only.split(",")))}


def ndcg10(rels, scores):
    order = np.argsort(-np.asarray(scores), kind="stable")[:10]
    dcg = sum((2 ** rels[i] - 1) / math.log2(k + 2) for k, i in enumerate(order))
    ideal = sorted(rels, reverse=True)[:10]
    idcg = sum((2 ** r - 1) / math.log2(k + 2) for k, r in enumerate(ideal))
    return dcg / idcg if idcg > 0 else 0.0


def mrr10(rels, scores):
    order = np.argsort(-np.asarray(scores), kind="stable")[:10]
    return next((1 / (k + 1) for k, i in enumerate(order) if rels[i] > 0), 0.0)


class CE:
    def __init__(self, spec):
        from transformers import AutoModelForSequenceClassification, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(spec, trust_remote_code=True)
        self.m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16, trust_remote_code=True).cuda().eval()

    @torch.no_grad()
    def __call__(self, pairs):
        out = np.zeros(len(pairs)); order = np.argsort([len(q) + len(d) for q, d in pairs])
        for s in range(0, len(pairs), args.bs):
            b = order[s:s + args.bs]
            enc = self.tok([pairs[i][0] for i in b], [pairs[i][1] for i in b], truncation="only_second", max_length=args.max_len,
                           padding=True, return_tensors="pt")
            lo = self.m(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits.float()
            out[b] = (lo[:, 0] if lo.shape[1] == 1 else lo[:, 1] - lo[:, 0]).cpu().numpy()
        return out


class Qwen3RR:
    PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
           "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
    SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    INST = "Given a web search query, retrieve relevant passages that answer the query"

    def __init__(self, spec):
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.tok = AutoTokenizer.from_pretrained(spec, padding_side="left")
        self.m = AutoModelForCausalLM.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
        self.yes, self.no = self.tok.convert_tokens_to_ids("yes"), self.tok.convert_tokens_to_ids("no")
        self.pre, self.suf = self.tok.encode(self.PRE, add_special_tokens=False), self.tok.encode(self.SUF, add_special_tokens=False)

    @torch.no_grad()
    def __call__(self, pairs):
        out = np.zeros(len(pairs)); order = np.argsort([len(q) + len(d) for q, d in pairs])
        for s in range(0, len(pairs), args.bs):
            b = order[s:s + args.bs]
            body = [self.tok.encode(f"<Instruct>: {self.INST}\n<Query>: {pairs[i][0]}\n<Document>: {pairs[i][1]}", add_special_tokens=False)
                    [: args.max_len - len(self.pre) - len(self.suf)] for i in b]
            ids = [self.pre + x + self.suf for x in body]; L = max(map(len, ids))
            inp = torch.full((len(ids), L), self.tok.pad_token_id); att = torch.zeros((len(ids), L), dtype=torch.long)
            for j, x in enumerate(ids):
                inp[j, L - len(x):] = torch.tensor(x); att[j, L - len(x):] = 1
            lo = self.m(input_ids=inp.cuda(), attention_mask=att.cuda(), logits_to_keep=1).logits[:, -1, :].float()
            out[b] = torch.log_softmax(torch.stack([lo[:, self.no], lo[:, self.yes]], 1), 1)[:, 1].exp().cpu().numpy()
        return out


res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    kind, mid = spec.split(":", 1) if spec.startswith("qwen3rr:") else ("ce", spec)
    model = Qwen3RR(mid) if kind == "qwen3rr" else CE(mid)
    r, t0, npairs = {}, time.time(), 0
    for name, df in EV.items():
        pairs = [(q, d) for q, ds in zip(df["query"], df.docs) for d in ds]; npairs += len(pairs)
        sc = model(pairs); k = 0; nd, mr = [], []
        for rels in df.rels:
            s = sc[k:k + len(rels)]; k += len(rels)
            nd.append(ndcg10(list(rels), s)); mr.append(mrr10(list(rels), s))
        r[name] = dict(n=len(df), ndcg10=float(np.mean(nd)), mrr10=float(np.mean(mr)))
    fam = lambda p: [v["ndcg10"] for k2, v in r.items() if k2.startswith(p)]
    r["_miracl"] = float(np.mean(fam("miracl_"))) if fam("miracl_") else None
    r["_wiki"] = float(np.mean(fam("wiki_"))) if fam("wiki_") else None
    oth = [v["ndcg10"] for k2, v in r.items() if not k2.startswith(("miracl_", "wiki_", "_"))]
    r["_other"] = float(np.mean(oth)) if oth else None
    r["_all"] = float(np.mean([x for x in (r["_miracl"], r["_wiki"], r["_other"]) if x is not None]))
    r["_pairs_per_s"] = npairs / (time.time() - t0)
    print(spec, {k2: (round(v, 4) if isinstance(v, float) else v) for k2, v in r.items() if k2.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
    del model; torch.cuda.empty_cache()
