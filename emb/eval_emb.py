"""Evaluate embedding models by cosine reranking on rerank/evals (nDCG@10, same sets/metric as rerank/eval_rerank.py).
python emb/eval_emb.py --models A,B --evals rerank/evals --out R.json
Specs: cls:<id> (CLS pooling, e.g. BAAI/bge-m3) | mean:<id> (mean pooling) | e5:<id> (mean pooling + "query: "/"passage: "
prefixes) | ours:<dir> (our student: mean pooling + dir/projection.pt) | asym:<dir> (queries by our student, documents by
bge-m3 - querying an existing bge-m3 index). All embeddings L2-normalised; documents truncated to 512 tokens.
"""
import argparse, glob, json, math, os, sys, time
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=128)
args = ap.parse_args()
EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}


def ndcg10(rels, scores):
    order = np.argsort(-np.asarray(scores), kind="stable")[:10]
    dcg = sum((2 ** rels[i] - 1) / math.log2(k + 2) for k, i in enumerate(order))
    idcg = sum((2 ** r - 1) / math.log2(k + 2) for k, r in enumerate(sorted(rels, reverse=True)[:10]))
    return dcg / idcg if idcg > 0 else 0.0


class Enc:
    def __init__(self, spec):
        from transformers import AutoModel, AutoTokenizer
        self.kind, mid = spec.split(":", 1)
        self.tok = AutoTokenizer.from_pretrained(mid); self.m = AutoModel.from_pretrained(mid, torch_dtype=torch.float16).cuda().eval()
        self.proj = None
        if self.kind == "ours":
            self.proj = torch.load(f"{mid}/projection.pt", map_location="cuda").half()

    @torch.no_grad()
    def __call__(self, texts, is_query):
        if self.kind == "e5":
            texts = [("query: " if is_query else "passage: ") + t for t in texts]
        out = np.zeros((len(texts), 0), dtype=np.float32); chunks = [None] * len(texts); order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), args.bs):
            b = order[s:s + args.bs]
            e = self.tok([texts[i] for i in b], truncation=True, max_length=64 if is_query else 512, padding=True, return_tensors="pt").to("cuda")
            h = self.m(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).last_hidden_state
            if self.kind == "cls":
                v = h[:, 0]
            else:
                a = e["attention_mask"].unsqueeze(-1).to(h.dtype); v = (h * a).sum(1) / a.sum(1)
            if self.proj is not None:
                v = v @ self.proj
            v = torch.nn.functional.normalize(v.float(), dim=-1).cpu().numpy()
            for j, i in enumerate(b):
                chunks[i] = v[j]
        return np.stack(chunks)


res = json.load(open(args.out)) if os.path.exists(args.out) else {}
bge = None
for spec in args.models.split(","):
    if spec.startswith("asym:"):
        bge = bge or Enc("cls:BAAI/bge-m3"); qenc, denc = Enc("ours:" + spec[5:]), bge
    else:
        qenc = denc = Enc(spec)
    r, t0, n = {}, time.time(), 0
    for name, df in EV.items():
        Q = qenc(df["query"].tolist(), True)
        docs = [d for ds in df.docs for d in ds]; D = denc(docs, False); n += len(docs)
        k = 0; nd = []
        for qi, rels in enumerate(df.rels):
            s = D[k:k + len(rels)] @ Q[qi]; k += len(rels); nd.append(ndcg10(list(rels), s))
        r[name] = dict(n=len(df), ndcg10=float(np.mean(nd)))
    fam = lambda p: [v["ndcg10"] for k2, v in r.items() if k2.startswith(p)]
    r["_miracl"], r["_wiki"] = float(np.mean(fam("miracl_"))), float(np.mean(fam("wiki_")))
    r["_other"] = float(np.mean([v["ndcg10"] for k2, v in r.items() if not k2.startswith(("miracl_", "wiki_", "_"))]))
    r["_all"] = float(np.mean([r["_miracl"], r["_wiki"], r["_other"]])); r["_docs_per_s"] = n / (time.time() - t0)
    print(spec, {k2: round(v, 4) for k2, v in r.items() if k2.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
    if not spec.startswith("asym:"):
        del qenc, denc
    torch.cuda.empty_cache()
