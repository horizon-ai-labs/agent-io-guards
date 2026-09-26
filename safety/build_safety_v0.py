"""Content-safety data v0 from nvidia/Nemotron-Safety-Guard-Dataset-v3 (CC-BY-4.0; Aegis 2.0 taxonomy, 12 languages).

python safety/build_safety_v0.py OUT     -> OUT/{train,val,test}.parquet (+ stats.json)
Columns: text (user prompt), text_pair (model response, "" for prompt-only items), unsafe (0/1), cats (list of our
category names), lang, kind ("prompt" | "response"), tag (generic / jailbreaking / adapted), id.
Each record gives up to two items:
  prompt item    text=prompt, unsafe=prompt_label
  response item  text=prompt, text_pair=response, unsafe=response_label   (only when response_label is set)
violated_categories are given per record; we attach them to the unsafe item(s) of that record (for records where both
prompt and response are unsafe the list may mix the two; documented as a limitation). REDACTED rows (Kaggle-derived)
are dropped. Splits follow the dataset's train / valid / test.
"""
import json, os, sys, collections
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
REPO = "nvidia/Nemotron-Safety-Guard-Dataset-v3"
LANGS = ["en", "ar", "de", "es", "fr", "hi", "it", "ja", "ko", "nl", "th", "zh"]
# Aegis 2.0 category -> our category (None = no category; still counted under "unsafe")
CAT = {"Hate/Identity Hate": "hate", "Harassment": "harassment", "Threat": "violence", "Violence": "violence",
       "Guns and Illegal Weapons": "weapons", "Sexual": "sexual", "Sexual (minor)": "sexual_minors",
       "Suicide and Self Harm": "self_harm", "Criminal Planning/Confessions": "criminal_planning",
       "Illegal Activity": "criminal_planning", "Malware": "criminal_planning", "Controlled/Regulated Substances": "drugs",
       "PII/Privacy": "privacy", "Profanity": "profanity", "Fraud/Deception": "fraud_manipulation",
       "Manipulation": "fraud_manipulation", "Political/Misinformation/Conspiracy": "misinformation",
       "Unauthorized Advice": "unauthorized_advice", "High Risk Gov Decision Making": "unauthorized_advice",
       "Immoral/Unethical": "other", "Copyright/Trademark/Plagiarism": "other", "Other": "other", "Needs Caution": None}
CATS = ["hate", "harassment", "violence", "weapons", "sexual", "sexual_minors", "self_harm", "criminal_planning", "drugs",
        "privacy", "profanity", "fraud_manipulation", "misinformation", "unauthorized_advice", "other"]
lab = lambda x: {"unsafe": 1, "safe": 0}.get((x or "").strip().lower())

stats = collections.defaultdict(collections.Counter)
for split, fn in [("train", "train"), ("val", "valid"), ("test", "test")]:
    rows = []
    for l in LANGS:
        for line in open(hf_hub_download(REPO, f"{l}/{fn}.jsonl", repo_type="dataset")):
            r = json.loads(line)
            if r["prompt"] == "REDACTED" or not (r["prompt"] or "").strip():
                stats[split]["dropped"] += 1; continue
            cats = sorted({CAT[c.strip()] for c in (r["violated_categories"] or "").split(",") if c.strip() and CAT.get(c.strip())})
            for c in (r["violated_categories"] or "").split(","):
                if c.strip() and c.strip() not in CAT:
                    stats[split]["unknown_cat:" + c.strip()] += 1
            p = lab(r["prompt_label"])
            if p is not None:
                rows.append(dict(text=r["prompt"], text_pair="", unsafe=p, cats=cats if p else [], lang=l, kind="prompt", tag=r["tag"], id=r["id"]))
            q = lab(r["response_label"])
            if q is not None and (r["response"] or "").strip():
                rows.append(dict(text=r["prompt"], text_pair=r["response"], unsafe=q, cats=cats if q else [], lang=l, kind="response", tag=r["tag"], id=r["id"]))
    df = pd.DataFrame(rows).drop_duplicates(subset=["text", "text_pair", "kind"])
    df.to_parquet(f"{OUT}/{split}.parquet")
    stats[split].update(n=len(df), unsafe=int(df.unsafe.sum()), prompt=int((df.kind == "prompt").sum()), response=int((df.kind == "response").sum()))
    for c in CATS:
        stats[split]["cat:" + c] = int(df.cats.map(lambda x: c in x).sum())
    print(split, dict(stats[split]), flush=True)
    print(df.groupby(["lang", "kind"]).unsafe.agg(["count", "mean"]).unstack().round(3).to_string(), flush=True)
tr_ids = set(pd.read_parquet(f"{OUT}/train.parquet").id); te_ids = set(pd.read_parquet(f"{OUT}/test.parquet").id)
stats["overlap"]["test_ids_in_train"] = len(te_ids & tr_ids)
print("test ids also in train:", len(te_ids & tr_ids), "of", len(te_ids))
json.dump(dict(categories=CATS, mapping=CAT, stats=stats), open(f"{OUT}/stats.json", "w"), indent=1)
