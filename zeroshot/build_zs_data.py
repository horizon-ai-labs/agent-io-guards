"""Build the NLI training set for the zero-shot classifier.  python zeroshot/build_zs_data.py OUT GEN_DIR [GEN_DIR ...]

Label 1 = entailment, 0 = not_entailment. Columns: text (premise), text_pair (hypothesis), label, source, lang.
Sources (all commercially usable):
  mnli / snli / wanli      English NLI (MultiNLI: OANC + CC-BY-SA parts; SNLI CC-BY-SA-4.0; WANLI CC-BY-4.0)
  tr_nli / tr_nli_xen      Qwen translations of MNLI/WANLI pairs; _xen keeps the English hypothesis (the zero-shot case)
  zs_passage / zs_short    Qwen-labelled FineWeb passages and Qwen-written short texts, turned into
                           (text, template.format(label)) pairs with the gold labels as positives
  zs_tax_passage / zs_tax_short / zs_aspect   (v2, zs2_*.jsonl) the same with generic label taxonomies
"""
import glob, json, os, random, sys
import pandas as pd
from datasets import load_dataset

OUT, GEN = sys.argv[1], sys.argv[2:]
# variants (env): SNLI_N (default 100000), SHORT_NEG (negatives per short text, default 3), ZS_REPEAT (default 1)
SNLI_N, SHORT_NEG, ZS_REPEAT = int(os.environ.get("SNLI_N", 100000)), int(os.environ.get("SHORT_NEG", 3)), int(os.environ.get("ZS_REPEAT", 1))
TAX_NEG = int(os.environ.get("TAX_NEG", 6))   # negatives per text from a generic taxonomy (v2 data)
rng = random.Random(0)
GENERIC = ["This example is {}.", "This text is about {}.", "The topic of this text is {}.", "This is about {}.", "{}",
           "This text is {}.", "This is an example of {}.", "It is about {}.", "The category is {}.", "This is {}."]
TASK_T = {"main topic": ["This text is about {}.", "The main topic is {}."],
          "genre / type of text": ["This text is a {}.", "This is a {}.", "The genre of this text is {}."],
          "intended audience": ["This text is written for {}.", "The intended audience is {}."],
          "overall sentiment or tone": ["The tone of this text is {}.", "This text is {}.", "The sentiment is {}."],
          "purpose of the text": ["The purpose of this text is {}.", "This text aims to {}."],
          "field or industry": ["This text is about {}.", "This text belongs to the field of {}."]}
rows = []
add = lambda t, h, l, s, lang: rows.append((t, h, int(l), s, lang))

# English NLI
L = {0: 1, 1: 0, 2: 0}
for r in load_dataset("nyu-mll/multi_nli", split="train"):
    if r["label"] in L:
        add(r["premise"], r["hypothesis"], L[r["label"]], "mnli", "English")
snli = [r for r in load_dataset("stanfordnlp/snli", split="train") if r["label"] in L]
for r in rng.sample(snli, SNLI_N):
    add(r["premise"], r["hypothesis"], L[r["label"]], "snli", "English")
for r in load_dataset("alisawuffles/WANLI", split="train"):
    add(r["premise"], r["hypothesis"], int(r["gold"] == "entailment"), "wanli", "English")
print("english nli", len(rows), flush=True)

# translated NLI
n0 = len(rows)
for g in GEN:
    for p in glob.glob(f"{g}/nli_tr_*.jsonl"):
        for line in open(p):
            j = json.loads(line); y = int(j["label"] == "entailment")
            add(j["premise"], j["hypothesis"], y, "tr_nli", j["lang"])
            add(j["premise"], j["hypothesis_en"], y, "tr_nli_xen", j["lang"])
print("translated nli", len(rows) - n0, flush=True)


def hyp(label, task, qwen_t):
    r = rng.random()
    if qwen_t and "{}" in qwen_t and r < 0.4:
        t = qwen_t
    elif task in TASK_T and r < 0.5:
        t = rng.choice(TASK_T[task])
    elif r < 0.65:
        t = "This example is {}."   # the transformers pipeline default
    else:
        t = rng.choice(GENERIC)
    lab = label.replace("_", " ") if "_" in label and rng.random() < 0.8 else label
    lab = lab.lower() if rng.random() < 0.5 else lab
    return t.replace("{}", lab, 1)


n0 = len(rows); seen = set()
for g in GEN:
    for p in glob.glob(f"{g}/zs_*.jsonl") + glob.glob(f"{g}/zs2_*.jsonl"):
        for line in open(p):
            j = json.loads(line)
            key = (j["text"].strip().lower(), j.get("task", "") if j["source"] == "tax_passage" else "")
            if key in seen or len(j["text"]) < 8:
                continue
            seen.add(key)
            gold = set(j["gold"]); neg = [l for l in j["labels"] if l not in gold and l.strip()]
            if j["source"] == "short":
                neg = rng.sample(neg, min(SHORT_NEG, len(neg)))
            elif j["source"].startswith("tax_"):
                neg = rng.sample(neg, min(TAX_NEG, len(neg)))
            src = "zs_" + j["source"]
            for _ in range(ZS_REPEAT):   # repeats get freshly sampled hypothesis templates
                for l in gold:
                    add(j["text"], hyp(l, j.get("task", ""), j.get("template", "")), 1, src, j["lang"])
                for l in neg:
                    add(j["text"], hyp(l, j.get("task", ""), j.get("template", "")), 0, src, j["lang"])
print("zero-shot synthetic", len(rows) - n0, flush=True)

df = pd.DataFrame(rows, columns=["text", "text_pair", "label", "source", "lang"])
# split by premise text so the same text never appears in both train and val
keys = list(df.text.unique()); rng.shuffle(keys); vk = set(keys[: int(0.01 * len(keys))])
va = df[df.text.isin(vk)]; tr = df[~df.text.isin(vk)].sample(frac=1.0, random_state=0)
os.makedirs(OUT, exist_ok=True)
tr.to_parquet(f"{OUT}/train.parquet"); va.to_parquet(f"{OUT}/val.parquet")
print("train", len(tr), "val", len(va), "pos rate", round(tr.label.mean(), 3))
print(tr.groupby("source").label.agg(["count", "mean"]))
print(tr.lang.value_counts().head(40))
