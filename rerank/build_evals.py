"""Reranking evaluation sets from MTEB reranking datasets (evaluation only).   python rerank/build_evals.py OUT_DIR
Each output parquet = one (dataset, language): rows qid, query, docs (candidate texts, in the dataset's candidate order),
rels (graded relevance per candidate). Queries are sampled (seed 0) up to CAP per set; all candidates of a query are kept.
Sets: MIRACL (18 languages, dev, CC-BY-SA-4.0), WikipediaRerankingMultilingual (CC-BY-SA-3.0), ESCI (product search,
Apache-2.0), RuBQ (ru), T2Reranking (zh), VoyageMMarco (ja), AskUbuntuDupQuestions and StackOverflowDupQuestions (en).
"""
import os, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = sys.argv[1]
os.makedirs(OUT, exist_ok=True)
api = HfApi()
SETS = [("mteb/MIRACLReranking", "miracl", 60), ("mteb/WikipediaRerankingMultilingual", "wiki", 60), ("mteb/ESCIReranking", "esci", 150),
        ("mteb/RuBQReranking", "rubq", 150), ("mteb/T2Reranking", "t2", 150), ("mteb/VoyageMMarcoReranking", "mmarco", 150),
        ("mteb/AskUbuntuDupQuestions", "askubuntu", 361), ("mteb/stackoverflowdupquestions-reranking", "stackoverflow", 300)]


def load(repo, files, part, split):
    cand = [f for f in files if (f.startswith(part + "/") or f.split("/")[0].endswith("-" + part)) and f.split("/")[-1].startswith(split)]
    return cand


for repo, name, cap in SETS:
    files = [s.rfilename for s in api.dataset_info(repo).siblings if s.rfilename.endswith(".parquet")]
    split = "test" if any("/test-" in f for f in files) else "dev"
    prefixes = sorted({f.split("/")[0].rsplit("-", 1)[0] for f in files if "-" in f.split("/")[0]}) or [""]
    for pre in prefixes:
        def part(p):
            alts = [f"{pre}-{p}/{split}" if pre else f"{p}/{split}"] + ([f"{pre}-qrels/{split}"] if p == "data" else [])
            for f in files:
                if any(f.startswith(a) for a in alts):
                    return pd.read_parquet(hf_hub_download(repo, f, repo_type="dataset"))
            return None
        corpus, queries, top = part("corpus"), part("queries"), part("top_ranked")
        qrels = part("qrels")
        if qrels is None:
            qrels = part("data")
        if corpus is None or queries is None or top is None or qrels is None:
            print(name, pre, "incomplete; skipped"); continue
        ctext = dict(zip(corpus["_id"], (corpus.get("title", "").fillna("") + " " + corpus.text).str.strip().str.slice(0, 2000)))
        qtext = dict(zip(queries["_id"], queries.text))
        rel = {(q, c): s for q, c, s in zip(qrels["query-id"], qrels["corpus-id"], qrels["score"])}
        top = top[top["query-id"].isin(qtext)].sample(frac=1.0, random_state=0)
        rows = []
        for q, cids in zip(top["query-id"], top["corpus-ids"]):
            cids = [c for c in cids if c in ctext]
            rels = [int(rel.get((q, c), 0)) for c in cids]
            if not any(r > 0 for r in rels) or all(r > 0 for r in rels):
                continue
            rows.append((q, qtext[q], [ctext[c] for c in cids], rels))
            if len(rows) >= cap:
                break
        tag = f"{name}_{pre}" if pre else name
        pd.DataFrame(rows, columns=["qid", "query", "docs", "rels"]).to_parquet(f"{OUT}/{tag}.parquet")
        print(tag, len(rows), "queries,", sum(len(r[2]) for r in rows), "pairs", flush=True)
