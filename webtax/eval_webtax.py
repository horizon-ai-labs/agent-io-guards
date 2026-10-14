"""Evaluate a topic & format classifier (48 sigmoid outputs topic_* / format_*) on webtax/evals: en_llama405b (WebOrganizer Llama-3.1-405B
labels, English) and fw2_judged (65 languages, Gemma-4-31B judge).   python webtax/eval_webtax.py --models A --out R.json"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
HERE = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, HERE)
from metrics import acc_f1
ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--out", required=True); args = ap.parse_args()
EV = {n: pd.read_parquet(os.path.join(HERE, "evals", f"{n}.parquet")) for n in ("en_llama405b", "fw2_judged")}
res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec); m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    labels = [m.config.id2label[i] for i in range(m.config.num_labels)]
    ti = [i for i, l in enumerate(labels) if l.startswith("topic_")]; fi = [i for i, l in enumerate(labels) if l.startswith("format_")]
    r = {}
    for name, df in EV.items():
        P = np.zeros((len(df), len(labels))); order = np.argsort(df.text.str.len().values)
        with torch.no_grad():
            for s in range(0, len(df), 32):
                b = order[s:s + 32]; e = tok(df.text.iloc[b].tolist(), truncation=True, max_length=512, padding=True, return_tensors="pt")
                P[b] = m(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).logits.float().cpu().numpy()
        pt = np.array([labels[ti[j]][6:] for j in P[:, ti].argmax(1)]); pf = np.array([labels[fi[j]][7:] for j in P[:, fi].argmax(1)])
        r[name] = dict(topic=acc_f1(pt, df.ref_topic.values, df.lang.values), format=acc_f1(pf, df.ref_format.values, df.lang.values))
        print(spec, name, {k: {kk: round(vv, 4) for kk, vv in v.items()} for k, v in r[name].items()}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
