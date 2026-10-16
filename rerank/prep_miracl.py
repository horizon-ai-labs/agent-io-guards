"""Real human queries for reranker v1.1: MIRACL train (hotchpotch/miracl-hf-unified, Apache-2.0; 40,203 queries in 16 languages with
judged relevant / non-relevant Wikipedia passages). Candidates per query = judged positives + judged negatives, padded to GROUP with random
passages of the same language corpus (easy negatives); teacher-scored later (rerank/teacher_score_vllm.py), as for the synthetic data.
The positives/negatives fields are row indices into the language's corpus (checked: the first queries' positive titles are printed).
MIRACL dev (our MIRACL eval) is a different split. CPU job.   python rerank/prep_miracl.py OUT_DIR   [GROUP=16, PID0=10000000]
-> OUT_DIR/miracl_q.parquet (qid, query, qtype, qlang, cands) and OUT_DIR/miracl_passages.parquet (pid, text, lang)"""
import os, random, sys
import numpy as np, pandas as pd, pyarrow.parquet as pq
from huggingface_hub import HfApi, hf_hub_download
R = "hotchpotch/miracl-hf-unified"; OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
G = int(os.environ.get("GROUP", 16)); PID0 = int(os.environ.get("PID0", 10_000_000)); rng = random.Random(0)
LANG = dict(ar="arb_Arab", bn="ben_Beng", en="eng_Latn", es="spa_Latn", fa="fas_Arab", fi="fin_Latn", fr="fra_Latn", hi="hin_Deva", id="ind_Latn",
            ja="jpn_Jpan", ko="kor_Hang", ru="rus_Cyrl", sw="swh_Latn", te="tel_Telu", th="tha_Thai", zh="zho_Hans")
files = HfApi().list_repo_files(R, repo_type="dataset"); Q, P, nxt = [], [], PID0
for l, code in LANG.items():
    qs = pd.read_parquet(hf_hub_download(R, f"{l}_queries/train-00000-of-00001.parquet", repo_type="dataset"))
    cf = sorted(f for f in files if f.startswith(f"{l}_corpus/")); sizes = [pq.ParquetFile(hf_hub_download(R, f, repo_type="dataset")).metadata.num_rows for f in cf]
    off = np.cumsum([0] + sizes); N = off[-1]
    need = {}
    for r in qs.itertuples():
        c = list(dict.fromkeys(list(r.positives) + list(r.negatives)))[:G]
        c += [rng.randrange(N) for _ in range(G - len(c))]
        need[r.query_id] = c
    rows = sorted({i for c in need.values() for i in c}); text = {}
    for k, f in enumerate(cf):
        sel = [i - off[k] for i in rows if off[k] <= i < off[k + 1]]
        if not sel:
            continue
        t = pq.read_table(hf_hub_download(R, f, repo_type="dataset"), columns=["title", "text"])
        ti, tx = t.column("title").to_pylist(), t.column("text").to_pylist()
        for i in sel:
            text[i + off[k]] = (ti[i] + "\n" + tx[i]) if ti[i] else tx[i]
        os.remove(hf_hub_download(R, f, repo_type="dataset")) if os.environ.get("CLEAN") else None
    pid = {}
    for i in rows:
        pid[i] = nxt; P.append(dict(pid=nxt, text=text[i][:3000], lang=code)); nxt += 1
    for r in qs.itertuples():
        Q.append(dict(qid=f"miracl-{l}-{r.query_id}", query=r.query, qtype="miracl", qlang=code, cands=np.array([pid[i] for i in need[r.query_id]])))
    r0 = qs.iloc[0]; print(l, len(qs), "queries", len(rows), "passages | check:", r0.query[:60], "->", text[int(r0.positives[0])][:80].replace("\n", " | "), flush=True)
pd.DataFrame(Q).to_parquet(f"{OUT}/miracl_q.parquet"); pd.DataFrame(P).to_parquet(f"{OUT}/miracl_passages.parquet"); print("total", len(Q), len(P))
