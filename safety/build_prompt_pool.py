"""Prompt / response pool for teacher labelling (content-safety data v1).  python safety/build_prompt_pool.py OUT.parquet

Why: v0 (Nemotron-Safety-Guard v3 = Aegis 2.0 + translations) is narrow; on PolyGuard and XSTest its ranking (AUC .82)
is far below Qwen3Guard-0.6B (.91 / .96). v1 adds diverse real and red-team prompts, labelled by Qwen3Guard-Gen-8B
(Apache-2.0) with soft scores (safety/label_teacher.py). Sources (all commercially usable):
  wildchat     allenai/WildChat-1M (ODC-BY): first user turn (+ first assistant reply as a response item), 4 of 14 files,
               all rows flagged toxic by the dataset plus a random sample of the others
  oasst2       OpenAssistant/oasst2 (Apache-2.0): initial prompts, 35 languages
  aya          CohereForAI/aya_dataset (Apache-2.0): human-written instructions, up to 800 per language
  salad        OpenSafetyLab/Salad-Data (Apache-2.0): base + attack-enhanced questions (ToxicChat-derived rows dropped)
  jbb          JailbreakBench/JBB-Behaviors (MIT): harmful + benign behaviours
Columns: text, text_pair ("" or response), kind, source, lang.
"""
import gzip, json, random, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]
rng = random.Random(0)
dl = lambda r, f: hf_hub_download(r, f, repo_type="dataset")
rows = []
add = lambda t, p, kind, src, lang: rows.append((t, p, kind, src, lang))
ok = lambda t: isinstance(t, str) and 5 <= len(t.strip()) <= 6000

# WildChat-1M
n_tox = n_non = 0
for k in [0, 3, 7, 11]:
    df = pd.read_parquet(dl("allenai/WildChat-1M", f"data/train-{k:05d}-of-00014.parquet"))
    if k == 0:
        print("wildchat columns", df.columns.tolist(), flush=True)
    for r in df.itertuples():
        conv = list(r.conversation)
        if not conv or conv[0]["role"] != "user" or not ok(conv[0]["content"]):
            continue
        tox = bool(getattr(r, "toxic", False))
        if not tox and rng.random() > 0.45:
            continue
        n_tox += tox; n_non += not tox
        lang = getattr(r, "language", "") or ""
        add(conv[0]["content"], "", "prompt", "wildchat", lang)
        if len(conv) > 1 and conv[1]["role"] == "assistant" and ok(conv[1]["content"]) and rng.random() < 0.5:
            add(conv[0]["content"], conv[1]["content"], "response", "wildchat", lang)
print("wildchat toxic", n_tox, "other", n_non, flush=True)

# oasst2 initial prompts
for line in gzip.open(dl("OpenAssistant/oasst2", "2023-11-05_oasst2_prompts.messages.jsonl.gz"), "rt"):
    j = json.loads(line)
    if ok(j.get("text")):
        add(j["text"], "", "prompt", "oasst2", j.get("lang", ""))

# Aya dataset, capped per language
a = pd.read_parquet(dl("CohereForAI/aya_dataset", "data/train-00000-of-00001.parquet"))
for lang, g in a.groupby("language"):
    for r in g.sample(n=min(800, len(g)), random_state=0).itertuples():
        if ok(r.inputs):
            add(r.inputs, "", "prompt", "aya", lang)

# Salad-Data
for f, key in [("base_set.json", "question"), ("attack_enhanced_set.json", "augq")]:
    for j in json.load(open(dl("OpenSafetyLab/Salad-Data", f))):
        if "toxicchat" in str(j.get("source", "")).lower():
            continue
        if ok(j.get(key)):
            add(j[key], "", "prompt", "salad", "English")

# JailbreakBench behaviours
for f in ["data/harmful-behaviors.csv", "data/benign-behaviors.csv"]:
    for g in pd.read_csv(dl("JailbreakBench/JBB-Behaviors", f)).Goal:
        if ok(g):
            add(g, "", "prompt", "jbb", "English")

df = pd.DataFrame(rows, columns=["text", "text_pair", "kind", "source", "lang"]).drop_duplicates(subset=["text", "text_pair"])
df = df.sample(frac=1.0, random_state=0).reset_index(drop=True)
df.to_parquet(OUT)
print(len(df)); print(df.groupby(["source", "kind"]).size()); print(df.lang.value_counts().head(30))
