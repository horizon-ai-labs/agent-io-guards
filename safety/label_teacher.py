"""Label a pool (safety/build_prompt_pool.py) with Qwen3Guard-Gen (default 8B).  python safety/label_teacher.py POOL OUT [MODEL]
Adds p_safe, p_unsafe, p_contro (next-token probabilities after "Safety:", see safety/qwenguard.py).
env SHARD / NSHARD: label only rows SHARD::NSHARD (row order kept via column "row")."""
import os, sys, time
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from qwenguard import QwenGuard

pool, out = sys.argv[1], sys.argv[2]
g = QwenGuard(sys.argv[3] if len(sys.argv) > 3 else "Qwen/Qwen3Guard-Gen-8B"); g.sanity()
df = pd.read_parquet(pool)
df["row"] = range(len(df))
sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
df = df.iloc[sh::ns].copy()
t0 = time.time()
pr = g.score(df.text.tolist(), df.text_pair.tolist(), log_every=20000)
df["p_safe"], df["p_unsafe"], df["p_contro"] = pr[:, 0], pr[:, 1], pr[:, 2]
df.to_parquet(out)
print(f"labelled {len(df)} in {time.time()-t0:.0f}s")
g = [c for c in ["source", "tag", "kind"] if c in df.columns]
print(df.assign(u=df.p_unsafe > 0.5, c=df.p_contro > 0.5).groupby(g)[["u", "c"]].mean().round(3))
