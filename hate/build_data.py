"""Hate-speech training data: teacher-labelled Civil pool (hate/build_pool.py) + teacher-labelled generated texts (hate/gen_hate.py).
v2 adds teacher-labelled real social-media texts (hate/build_pool2.py) with --real SHARDS..., repeated REAL_MULT times (default 2) in train.
Val = 5% held out: Civil by source comment (a comment and its translations stay together), generated and real texts at random.
python hate/build_data.py OUT_DIR TOX_V0_TRAIN.parquet --civil SHARDS... --gen SHARDS... [--real SHARDS...]"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
L = ["hate", "target_race", "target_religion", "target_origin", "target_gender", "target_sexuality", "target_age", "target_disability"]
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text}
out, v0 = sys.argv[1:3]; a = sys.argv[3:]; ci, gi = a.index("--civil"), a.index("--gen"); ri = a.index("--real") if "--real" in a else len(a)
civ = pd.concat([pd.read_parquet(p) for p in a[ci + 1:gi]], ignore_index=True)
gen = pd.concat([pd.read_parquet(p) for p in a[gi + 1:ri]], ignore_index=True).assign(source="gen")
real = pd.concat([pd.read_parquet(p) for p in a[ri + 1:]], ignore_index=True) if ri < len(a) else pd.DataFrame(columns=["text", "lang", "source"] + L)
src = pd.read_parquet(v0, columns=["text", "src_idx"]).drop_duplicates("text").set_index("text").src_idx
civ["grp"] = civ.text.map(src).fillna(-1).astype(int)
rng = np.random.RandomState(0)
vg = set(rng.choice(civ.grp.unique(), int(0.05 * civ.grp.nunique()), replace=False))
civ["val"] = civ.grp.isin(vg); gen["val"] = rng.rand(len(gen)) < 0.05; real["val"] = rng.rand(len(real)) < 0.05
d = pd.concat([civ, gen, real], ignore_index=True)
k = d.text.map(norm); n_ev = int(k.isin(ev).sum()); d = d[~k.isin(ev) & (k.str.len() >= 3)].drop_duplicates("text")
cols = ["text", "lang", "source"] + L
os.makedirs(out, exist_ok=True)
tr, va = d[~d.val][cols], d[d.val][cols]
real_tr = tr[tr.source.isin(["mhs", "hatexplain"])]
tr = pd.concat([tr] + [real_tr] * (int(os.environ.get("REAL_MULT", 2)) - 1)).sample(frac=1, random_state=0)
tr.to_parquet(f"{out}/train.parquet"); va.to_parquet(f"{out}/val.parquet")
st = dict(train=len(tr), val=len(va), real_mult=int(os.environ.get("REAL_MULT", 2)) if len(real) else 0, removed_eval_overlap=n_ev, by_source=d.source.value_counts().to_dict(),
          hate_ge_05={s: round(float((g.hate >= 0.5).mean()), 3) for s, g in d.groupby("source")},
          target_ge_05_among_hate={l: round(float((d[d.hate >= 0.5][l] >= 0.5).mean()), 3) for l in L[1:]},
          gen_hate_by_intended={k: round(float(v), 3) for k, v in d[d.source == "gen"].groupby("intended").hate.mean().items() if k in ("hateful", "non-hateful", "offensive")} if "intended" in d else {},
          n_lang=int(d.lang.nunique()))
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
