"""Educational-quality training data.
- Teacher (Qwen3.8-27B, edu/teacher_edu.py) scores for FineWeb-2 documents, mapped onto the Llama-3 FineWeb-Edu scale with an isotonic
  (PAV) fit on 10,000 English documents scored by both (source 'llama_calib' in the label shards);
- English: N_EN documents from shards 0-6 of HuggingFaceFW/fineweb-edu-llama3-annotations with their Llama-3 scores (shard 7 = eval).
Val = 5% random. Label column 'edu' (float, 0-5).   python edu/build_data.py OUT_DIR LABEL_SHARDS...   [N_EN=150000]"""
import glob, json, os, re, sys
import numpy as np, pandas as pd
from huggingface_hub import hf_hub_download

out = sys.argv[1]; lab = pd.concat([pd.read_parquet(p) for p in sys.argv[2:]], ignore_index=True)


def pav(x, y):
    o = np.argsort(x, kind="stable"); blocks = []
    for a, b in zip(x[o], y[o]):
        blocks.append([b, 1, a])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] > blocks[-1][0] / blocks[-1][1]:
            s, c, m = blocks.pop(); blocks[-1][0] += s; blocks[-1][1] += c; blocks[-1][2] = m
    return np.array([b[2] for b in blocks]), np.array([b[0] / b[1] for b in blocks])


cal = lab[(lab.source == "llama_calib") & lab.edu.notna()]; xm, ym = pav(cal.edu.values, cal.llama.values.astype(float))
f = lambda s: ym[np.minimum(np.searchsorted(xm, s, side="left"), len(xm) - 1)]
fw = lab[(lab.source == "fineweb2") & lab.edu.notna()].copy(); fw["edu_raw"] = fw.edu; fw["edu"] = f(fw.edu.values)
rk = lambda a: pd.Series(a).rank().values
calib = dict(n=len(cal), spearman_teacher_llama=float(np.corrcoef(rk(cal.edu), rk(cal.llama))[0, 1]),
             map={str(q): round(float(f(np.array([q]))[0]), 3) for q in (0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5)})
N = int(os.environ.get("N_EN", 150000)); calib_texts = set(cal.text)
en = pd.concat([pd.read_parquet(hf_hub_download("HuggingFaceFW/fineweb-edu-llama3-annotations", f"data/train-0000{i}-of-00008.parquet", repo_type="dataset"),
                                columns=["text", "score"]) for i in range(7)], ignore_index=True)
en = en.sample(N, random_state=0).assign(text=lambda d: d.text.str[:6000], lang="eng_Latn", source="llama3", edu=lambda d: d.score.astype(float))
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()[:500]
ev = {norm(t) for p in glob.glob(os.path.join(os.path.dirname(os.path.abspath(__file__)), "evals", "*.parquet")) for t in pd.read_parquet(p).text} - {""}
d = pd.concat([fw[["text", "lang", "source", "edu"]], en[["text", "lang", "source", "edu"]]], ignore_index=True)
k = d.text.map(norm); n_ev = int(k.isin(ev).sum()); d = d[~k.isin(ev)].drop_duplicates("text")
d["val"] = np.random.RandomState(0).rand(len(d)) < 0.05
os.makedirs(out, exist_ok=True); cols = ["text", "lang", "source", "edu"]
d[~d.val][cols].sample(frac=1, random_state=0).to_parquet(f"{out}/train.parquet"); d[d.val][cols].to_parquet(f"{out}/val.parquet")
st = dict(train=int((~d.val).sum()), val=int(d.val.sum()), removed_eval_overlap=n_ev, by_source=d.source.value_counts().to_dict(), n_lang=int(d.lang.nunique()),
          calibration=calib, mean_edu={s: round(float(g.edu.mean()), 3) for s, g in d.groupby("source")},
          frac_ge3={s: round(float((g.edu >= 2.5).mean()), 3) for s, g in d.groupby("source")})
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st, indent=1))
