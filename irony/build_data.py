"""Irony training data: teacher-labelled generated texts (irony/build_gen.py) + teacher-labelled Civil pool (irony/build_pool.py).
Val = 5% held out: Civil by source comment (a comment and its translations together), generated texts by post (a thread together).
python irony/build_data.py OUT_DIR --civil SHARDS... --gen SHARDS..."""
import hashlib, json, os, sys
import numpy as np, pandas as pd
out = sys.argv[1]; a = sys.argv[2:]; ci, gi = a.index("--civil"), a.index("--gen")
civ = pd.concat([pd.read_parquet(p) for p in a[ci + 1:gi]], ignore_index=True)
gen = pd.concat([pd.read_parquet(p) for p in a[gi + 1:]], ignore_index=True)
rng = np.random.RandomState(0)
hsh = lambda t: hashlib.md5(t.encode()).hexdigest()[:12]  # stable across runs (unlike hash())
civ["grp"] = "c" + civ.src_idx.astype(str); gen["grp"] = np.where(gen.context != "", "p" + gen.context.map(hsh), "g" + gen.index.astype(str))
tg = dict(zip(gen.text[gen.context != ""], gen.grp[gen.context != ""]))  # a reply's no-context copy stays with its thread
gen.loc[gen.context == "", "grp"] = [tg.get(t, "t" + hsh(t)) for t in gen.text[gen.context == ""]]
d = pd.concat([civ, gen], ignore_index=True); g = d.grp.unique()
vg = set(rng.choice(g, int(0.05 * len(g)), replace=False)); d["val"] = d.grp.isin(vg)
cols = ["text", "context", "lang", "source", "irony"]
os.makedirs(out, exist_ok=True)
tr, va = d[~d.val][cols].sample(frac=1, random_state=0), d[d.val][cols]
tr.to_parquet(f"{out}/train.parquet"); va.to_parquet(f"{out}/val.parquet")
st = dict(train=len(tr), val=len(va), by_source=d.source.value_counts().to_dict(), with_context=int((d.context != "").sum()),
          irony_ge_05={s: round(float((x.irony >= 0.5).mean()), 3) for s, x in d.groupby("source")},
          gen_irony_by_intended={k: round(float(v), 3) for k, v in gen.groupby("intended").irony.mean().items() if k in ("ironic", "sincere", "humorous")},
          gen_irony_ctx_vs_noctx=dict(ctx=round(float(gen[gen.context != ""].irony.mean()), 3), noctx=round(float(gen[gen.context == ""].irony.mean()), 3)),
          n_lang=int(d.lang.nunique()))
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
