"""Content-safety data v1 = v0 (Nemotron-Safety-Guard v3, hard labels + categories) + teacher-labelled pool.

python safety/build_safety_v1.py V0_DIR POOL_LABELLED EVAL_DIR OUT [POOL_FRAC]
Pool rows (safety/build_prompt_pool.py, labelled by safety/label_teacher.py with Qwen3Guard-Gen-8B) get
soft = P(unsafe) + 0.5 * P(controversial), has_cats=False (category loss masked). Pool texts that also occur in any
benchmark (safety/evals, normalised exact match on prompt or response) are removed. val/test stay v0 (hard labels).
"""
import glob, json, os, re, shutil, sys
import numpy as np, pandas as pd

V0, POOL, EV, OUT = sys.argv[1:5]
FRAC = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())

bench = set()
for p in glob.glob(f"{EV}/*.parquet"):
    d = pd.read_parquet(p)
    bench |= {norm(x) for x in d.text} | {norm(x) for x in d.text_pair if x}
tr = pd.read_parquet(f"{V0}/train.parquet")
bench_hits_v0 = int(tr.text.map(norm).isin(bench).sum())

pool = pd.read_parquet(POOL)
hit = pool.text.map(norm).isin(bench) | pool.text_pair.map(lambda x: bool(x) and norm(x) in bench)
print("pool", len(pool), "removed as benchmark overlap", int(hit.sum()), "| v0 train rows matching a benchmark text:", bench_hits_v0, flush=True)
pool = pool[~hit]
if FRAC < 1:
    pool = pool.sample(frac=FRAC, random_state=0)
soft = (pool.p_unsafe + 0.5 * pool.p_contro).clip(0, 1).values
pl = pd.DataFrame(dict(text=pool.text.values, text_pair=pool.text_pair.values, unsafe=(soft > 0.5).astype(int), soft=soft,
                       cats=[[] for _ in range(len(pool))], has_cats=False, lang=pool.lang.values, kind=pool.kind.values,
                       tag=pool.source.values, id=""))
tr = tr.assign(soft=np.nan, has_cats=True)
out = pd.concat([tr, pl], ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{OUT}/train.parquet")
for f in ["val.parquet", "test.parquet", "stats.json"]:
    shutil.copy(f"{V0}/{f}", f"{OUT}/{f}")
print("train", len(out), "(v0", len(tr), "+ pool", len(pl), ") unsafe rate", round(out.unsafe.mean(), 3))
print(pl.groupby(["tag", "kind"]).agg(n=("soft", "size"), soft_mean=("soft", "mean"), unsafe=("unsafe", "mean")).round(3))
json.dump(dict(pool=len(pl), removed_overlap=int(hit.sum()), v0_bench_hits=bench_hits_v0), open(f"{OUT}/v1_stats.json", "w"))
