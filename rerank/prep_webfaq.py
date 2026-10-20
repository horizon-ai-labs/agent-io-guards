"""Real FAQ questions for reranker v1.2: PaDaS-Lab/webfaq (CC-BY-4.0; question/answer pairs from schema.org FAQ markup on websites),
for the 45 of our 47 reranker languages it covers. Streams only the first lines of each language file (HfFileSystem range reads).
Filters: question 10-300 and answer 40-2000 characters, no spam/gambling/adult domains, <= 50 pairs per site, unique questions and
answers. Per language: a pool of POOL answers (the mining corpus) and QPL queries whose answer is in the pool (qtype 'faq').
CPU job.   python rerank/prep_webfaq.py OUT_DIR  [POOL=30000 QPL=2500 PID0=20000000]
-> OUT_DIR/webfaq_passages.parquet (pid, text, lang) and OUT_DIR/q/q_webfaq.jsonl (pid, query, qtype, qlang)"""
import json, os, random, re, sys, time
from collections import Counter
import pandas as pd
from huggingface_hub import HfFileSystem
OUT = sys.argv[1]; POOL, QPL, PID0 = (int(os.environ.get(k, d)) for k, d in (("POOL", 30000), ("QPL", 2500), ("PID0", 20_000_000)))
ALIAS = {"arb": "ara", "cmn": "zho", "ekk": "est", "fil": "tgl", "lvs": "lav", "nob": "nor", "zsm": "msa"}
OURS = ['arb_Arab', 'ben_Beng', 'bul_Cyrl', 'cat_Latn', 'ces_Latn', 'cmn_Hani', 'dan_Latn', 'deu_Latn', 'ekk_Latn', 'ell_Grek', 'eng_Latn', 'fas_Arab',
        'fil_Latn', 'fin_Latn', 'fra_Latn', 'heb_Hebr', 'hin_Deva', 'hrv_Latn', 'hun_Latn', 'ind_Latn', 'ita_Latn', 'jpn_Jpan', 'kor_Hang', 'lit_Latn',
        'lvs_Latn', 'mar_Deva', 'nld_Latn', 'nob_Latn', 'pol_Latn', 'por_Latn', 'ron_Latn', 'rus_Cyrl', 'slk_Latn', 'slv_Latn', 'spa_Latn', 'srp_Cyrl',
        'swe_Latn', 'tam_Taml', 'tel_Telu', 'tha_Thai', 'tur_Latn', 'ukr_Cyrl', 'urd_Arab', 'vie_Latn']
SPAM = re.compile(r"casino|1win|1xbet|bet(ting)?|poker|slot|porn|sex|escort|xxx|viagra|cialis|loan|payday|crypto|forex|gambl|lottery|dating", re.I)
fs = HfFileSystem(); rng = random.Random(0); P, Q, nxt = [], [], PID0
os.makedirs(f"{OUT}/q", exist_ok=True)
for code in OURS:
    w = ALIAS.get(code[:3], code[:3]); pairs, seen_q, seen_a, per_site, n = [], set(), set(), Counter(), 0
    for k in range(5):
        try:
            with fs.open(f"datasets/PaDaS-Lab/webfaq/data/{w}.jsonl", "r", block_size=8 << 20) as f:
                for line in f:
                    n += 1
                    r = json.loads(line); q, a, o = r.get("question", "").strip(), r.get("answer", "").strip(), r.get("origin", "")
                    if not (10 <= len(q) <= 300 and 40 <= len(a) <= 2000) or SPAM.search(o) or per_site[o] >= 50 or q in seen_q or a in seen_a:
                        continue
                    per_site[o] += 1; seen_q.add(q); seen_a.add(a); pairs.append((q, a))
                    if len(pairs) >= POOL or n > 40 * POOL:
                        break
            break
        except Exception as e:
            print(code, "retry", k, repr(e)[:120], flush=True); time.sleep(30 * (k + 1)); pairs, seen_q, seen_a, per_site, n = [], set(), set(), Counter(), 0
    ids = list(range(nxt, nxt + len(pairs))); nxt += len(pairs)
    P += [dict(pid=i, text=a, lang=code) for i, (_, a) in zip(ids, pairs)]
    qi = rng.sample(range(len(pairs)), min(QPL, len(pairs)))
    Q += [dict(pid=ids[i], query=pairs[i][0], qtype="faq", qlang=code) for i in qi]
    print(code, w, "lines", n, "pairs", len(pairs), "sites", len(per_site), "queries", len(qi), flush=True)
pd.DataFrame(P).to_parquet(f"{OUT}/webfaq_passages.parquet")
with open(f"{OUT}/q/q_webfaq.jsonl", "w") as f:
    for r in Q:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print("total passages", len(P), "queries", len(Q))
