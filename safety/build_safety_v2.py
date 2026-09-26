"""Content-safety data v2 = teacher-relabelled v0 + pool 1 (safety/build_prompt_pool.py) + pool 2 (safety/build_pool2.py).

python safety/build_safety_v2.py V0_DIR V0_LABELLED_1,V0_LABELLED_2,... POOL1 POOL2 EVAL_DIR OUT
env ALPHA (default 1.0): unsafe target of v0 rows = ALPHA * teacher + (1 - ALPHA) * human label; categories stay human.
Teacher score = P(unsafe) + 0.5 * P(controversial) from Qwen3Guard-Gen-8B (safety/label_teacher.py, sharded).
Why: v1 mixed human Aegis labels (stricter: sensitive-but-benign prompts marked unsafe) with teacher labels; XSTest FPR
stayed .34 and PolyGuard prompt AUC .83. Pool rows overlapping any benchmark are removed (as in v1).
"""
import glob, json, os, re, shutil, sys
import numpy as np, pandas as pd

V0, V0L, P1, P2, EV, OUT = sys.argv[1:7]
ALPHA = float(os.environ.get("ALPHA", 1.0))
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
teacher = lambda d: (d.p_unsafe + 0.5 * d.p_contro).clip(0, 1).values

bench = set()
for p in glob.glob(f"{EV}/*.parquet"):
    d = pd.read_parquet(p)
    bench |= {norm(x) for x in d.text} | {norm(x) for x in d.text_pair if x}

v0 = pd.concat([pd.read_parquet(f) for f in V0L.split(",")]).sort_values("row").reset_index(drop=True)
assert len(v0) == len(pd.read_parquet(f"{V0}/train.parquet")), "missing v0 shards"
t = teacher(v0)
print("v0 rows", len(v0), "teacher unsafe rate", round((t > 0.5).mean(), 3), "human unsafe rate", round(v0.unsafe.mean(), 3),
      "agreement", round(((t > 0.5) == (v0.unsafe == 1)).mean(), 3), flush=True)
print(v0.assign(t=t > 0.5).groupby(["kind", "unsafe"]).t.mean().round(3).to_string(), flush=True)
v0 = v0.assign(soft=ALPHA * t + (1 - ALPHA) * v0.unsafe.values, has_cats=True)
v0 = v0.assign(unsafe=(v0.soft > 0.5).astype(int))

parts = [v0.drop(columns=[c for c in ["row", "p_safe", "p_unsafe", "p_contro"] if c in v0.columns])]
for P in [P1, P2]:
    pool = pd.read_parquet(P)
    hit = pool.text.map(norm).isin(bench) | pool.text_pair.map(lambda x: bool(x) and norm(x) in bench)
    pool = pool[~hit]; s = teacher(pool)
    print(os.path.basename(P), "rows", len(pool), "removed overlap", int(hit.sum()), flush=True)
    parts.append(pd.DataFrame(dict(text=pool.text.values, text_pair=pool.text_pair.values, unsafe=(s > 0.5).astype(int), soft=s,
                                   cats=[[] for _ in range(len(pool))], has_cats=False, lang=pool.lang.values, kind=pool.kind.values,
                                   tag=pool.source.values, id="")))
out = pd.concat(parts, ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{OUT}/train.parquet")
for f in ["val.parquet", "test.parquet", "stats.json"]:
    shutil.copy(f"{V0}/{f}", f"{OUT}/{f}")
print("train", len(out), "unsafe rate", round(out.unsafe.mean(), 3), "ALPHA", ALPHA)
print(out.groupby("tag").agg(n=("soft", "size"), soft=("soft", "mean")).round(3).to_string())
