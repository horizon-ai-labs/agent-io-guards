"""Hard-negative candidates with BAAI/bge-m3 dense retrieval (MIT).   python rerank/mine.py PASSAGES.parquet QDIR OUT.parquet [K]
For each generated query, the top-K passages (cosine, CLS-pooled normalised bge-m3 embeddings) from the passages in the
same language as the query's source passage; the source passage is always included. OUT rows: qid, pid (source),
query, qtype, qlang, cands (list of K pids, source first).
"""
import glob, os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

P, QD, OUT = sys.argv[1:4]; K = int(sys.argv[4]) if len(sys.argv) > 4 else 24
ps = pd.read_parquet(P)
qs = pd.concat([pd.read_json(f, lines=True) for f in sorted(glob.glob(f"{QD}/*.jsonl"))], ignore_index=True)
qs = qs.drop_duplicates(["pid", "query"]).reset_index(drop=True); qs["qid"] = range(len(qs))
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); m = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()


@torch.no_grad()
def emb(texts, L, bs=256):
    out = np.zeros((len(texts), 1024), dtype=np.float16); order = np.argsort([len(t) for t in texts])
    for s in range(0, len(texts), bs):
        b = order[s:s + bs]
        e = tok([texts[i] for i in b], truncation=True, max_length=L, padding=True, return_tensors="pt").to("cuda")
        h = m(**e).last_hidden_state[:, 0]; out[b] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu().numpy()
    return out


PE = emb(ps.text.tolist(), 512); QE = emb(qs["query"].tolist(), 64)
plang = dict(zip(ps.pid, ps.lang)); pidx = {p: i for i, p in enumerate(ps.pid)}
qs["plang"] = qs.pid.map(plang)
cands = [None] * len(qs)
for lang, g in qs.groupby("plang"):
    pool = np.where(ps.lang.values == lang)[0]
    A = torch.tensor(PE[pool]).cuda()
    for s in range(0, len(g), 4096):
        gi = g.index.values[s:s + 4096]
        sim = torch.tensor(QE[gi]).cuda() @ A.T
        top = sim.topk(min(K + 1, len(pool)), dim=1).indices.cpu().numpy()
        for row, t in zip(gi, top):
            src = qs.pid[row]; c = [int(ps.pid.values[pool[j]]) for j in t if ps.pid.values[pool[j]] != src][: K - 1]
            cands[row] = [int(src)] + c
    print(lang, len(g), flush=True)
qs["cands"] = cands
qs[["qid", "pid", "query", "qtype", "qlang", "cands"]].to_parquet(OUT); print("queries", len(qs), "pairs", sum(map(len, cands)), flush=True)
