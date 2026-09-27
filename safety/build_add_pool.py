"""Add a teacher-labelled pool to an existing content-safety training set.  python safety/build_add_pool.py BASE_DIR POOL EVAL_DIR OUT
Pool rows: soft = P(unsafe) + 0.5 P(controversial) (Qwen3Guard-Gen-8B), category loss masked; benchmark-text overlaps removed.
Used for v5 = v4 + synthetic benign-but-alarming prompts and harmful look-alikes (safety/gen_edgy.py)."""
import glob, os, re, shutil, sys
import pandas as pd

B, P, EV, OUT = sys.argv[1:5]
os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", str(s).strip().lower())
bench = set()
for p in glob.glob(f"{EV}/*.parquet"):
    bench |= {norm(x) for x in pd.read_parquet(p).text}
pool = pd.read_parquet(P)
hit = pool.text.map(norm).isin(bench); pool = pool[~hit]
s = (pool.p_unsafe + 0.5 * pool.p_contro).clip(0, 1).values
print("pool", len(pool), "removed overlap", int(hit.sum()))
if "intended" in pool:
    print("teacher unsafe rate by generator intent:", pool.assign(u=s > 0.5).groupby("intended").u.mean().round(3).to_dict())
add = pd.DataFrame(dict(text=pool.text.values, text_pair="", unsafe=(s > 0.5).astype(int), soft=s, cats=[[] for _ in range(len(pool))],
                        has_cats=False, lang=pool.lang.values, kind="prompt", tag=pool.source.values, id=""))
out = pd.concat([pd.read_parquet(f"{B}/train.parquet"), add], ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{OUT}/train.parquet")
for f in ["val.parquet", "test.parquet", "stats.json"]:
    shutil.copy(f"{B}/{f}", f"{OUT}/{f}")
print("train", len(out))
