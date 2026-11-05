"""False alarms of the prompt-injection guards on REAL benign multilingual user prompts (never used in training):
OASST2 prompter turns (Apache-2.0) and Aya dataset prompts (Apache-2.0, human-written). Prompts are split by hash: half 'eval'
(scored here), half 'train' (kept for a possible v3; never scored). Flag rate per language at thresholds 0.5 / 0.9.
python inj/benign_multi_fpr.py OUT_DIR MODEL_DIR [MODEL_DIR ...]   -> OUT_DIR/benign_multi.parquet, fpr.json, flagged.jsonl"""
import hashlib, json, os, sys
import numpy as np, pandas as pd, torch
from datasets import load_dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

out, models = sys.argv[1], sys.argv[2:]
os.makedirs(out, exist_ok=True)
PER_LANG, MIN_N = 400, 50
rows = []
for r in load_dataset("OpenAssistant/oasst2", split="train"):
    if r["role"] == "prompter" and r["lang"]:
        rows.append(dict(text=r["text"], lang=r["lang"].split("-")[0].lower(), src="oasst2", parent=r["parent_id"] is None))
for r in load_dataset("CohereLabs/aya_dataset", split="train"):
    rows.append(dict(text=r["inputs"], lang=r["language_code"], src="aya", parent=True))
d = pd.DataFrame(rows)
d["text"] = d.text.str.strip(); d = d[(d.text.str.len() >= 10) & (d.text.str.len() <= 3000)].drop_duplicates("text")
d["split"] = ["eval" if int(hashlib.md5(t.encode()).hexdigest(), 16) % 2 == 0 else "train" for t in d.text]
d.to_parquet(f"{out}/benign_multi.parquet", index=False)
ev = d[d.split == "eval"]
ev = ev.groupby(["lang", "src"], group_keys=False).apply(lambda g: g.sample(min(len(g), PER_LANG), random_state=0))
keep = ev.lang.value_counts(); ev = ev[ev.lang.isin(keep[keep >= MIN_N].index)].reset_index(drop=True)
print("all", len(d), "eval sample", len(ev), "langs", ev.lang.nunique(), flush=True)

res, flagged = {}, []
for m in models:
    name = os.path.basename(os.path.dirname(m.rstrip("/")))[:8]
    tok = AutoTokenizer.from_pretrained(m); model = AutoModelForSequenceClassification.from_pretrained(m).cuda().eval()
    inj = [i for i, l in model.config.id2label.items() if l.upper() != "SAFE"][0]
    texts = ev.text.tolist(); order = np.argsort([len(t) for t in texts]); p = np.zeros(len(texts))
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        with torch.no_grad():
            p[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, int(inj)].cpu().numpy()
    ev[name] = p
    per = ev.groupby("lang").agg(n=(name, "size"), fpr50=(name, lambda x: float((x >= .5).mean())), fpr90=(name, lambda x: float((x >= .9).mean())))
    bysrc = ev.groupby("src").agg(n=(name, "size"), fpr50=(name, lambda x: float((x >= .5).mean())), fpr90=(name, lambda x: float((x >= .9).mean())))
    res[name] = dict(model=m.split("/")[-2], overall=dict(fpr50=float((p >= .5).mean()), fpr90=float((p >= .9).mean())),
                     lang_mean=dict(fpr50=float(per.fpr50.mean()), fpr90=float(per.fpr90.mean())),
                     by_src=bysrc.round(4).to_dict("index"), per_lang=per.round(4).to_dict("index"))
    print(name, res[name]["overall"], res[name]["lang_mean"], res[name]["by_src"], flush=True)
    print(" worst fpr50:", per.sort_values("fpr50", ascending=False).head(10).round(3).to_dict("index"), flush=True)
    for i in np.argsort(-p)[:150]:
        flagged.append(dict(model=name, p=round(float(p[i]), 4), lang=ev.lang[i], src=ev.src[i], text=ev.text[i][:500]))
json.dump(res, open(f"{out}/fpr.json", "w"), indent=1)
with open(f"{out}/flagged.jsonl", "w") as f:
    for r in flagged:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
ev.to_parquet(f"{out}/scored_eval.parquet", index=False)
