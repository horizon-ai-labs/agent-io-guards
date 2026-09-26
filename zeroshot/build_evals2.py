"""Extra zero-shot benchmarks (eval only, never trained on).  .venv/bin/python zeroshot/build_evals2.py OUTDIR

Why: MASSIVE and Banking77 were our only label sets that are unseen in training, so every release decision leaned on
them. Label-name overlap with our English synthetic labels (seen >= 3 times): MTOP 2/88, CLINC150 31/150 (so CLINC is
marked § in cards), MASSIVE 1/59, Banking77 2/77.
  mtop   MTOP intents (Facebook, CC-BY-SA-4.0; via mteb/mtop_intent), 6 languages, 300 test items each, all intents that
         occur in those test sets
  clinc  CLINC150 intents (CC-BY-3.0), "plus" test without out-of-scope, 1000 items, 150 intents
Same format as zeroshot/build_evals.py; labels keep the dataset's names with "_" -> " " (as for MASSIVE).
"""
import json, os, random, re, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
dl = lambda repo, f: hf_hub_download(repo, f, repo_type="dataset")


def save(name, template, labels, items):
    json.dump(dict(template=template, labels=labels, items=items), open(f"{OUT}/{name}.json", "w"), ensure_ascii=False)
    print(name, len(labels), "labels", len(items), "items", flush=True)


def sample(rows, n):
    rows = list(rows); rng.shuffle(rows); return rows[:n]


dfs = {l: pd.read_parquet(dl("mteb/mtop_intent", f"{l}/test-00000-of-00001.parquet")) for l in ["en", "de", "es", "fr", "hi", "th"]}
labels = sorted(set().union(*[set(d.label_text) for d in dfs.values()]))
items = [dict(text=r.text, gold=labels.index(r.label_text), lang=l) for l, d in dfs.items() for r in sample(d.itertuples(), 300)]
save("mtop", "This request is about {}.", [x.lower().replace("_", " ") for x in labels], items)

t = open(dl("clinc/clinc_oos", "README.md")).read()
i = t.find("config_name: plus"); j = t.find("splits:", i)
names = {int(k): v.strip().strip("'") for k, v in re.findall(r"'(\d+)': (.+)", t[i:j])}
assert len(names) == 151
df = pd.read_parquet(dl("clinc/clinc_oos", "plus/test-00000-of-00001.parquet"))
df = df[df.intent.map(names) != "oos"]
keep = sorted(v for v in names.values() if v != "oos")
save("clinc", "This request is about {}.", [x.replace("_", " ") for x in keep],
     [dict(text=r.text, gold=keep.index(names[r.intent]), lang="en") for r in sample(df.itertuples(), 1000)])
