"""Evaluate a prompt task & complexity classifier (17 sigmoid outputs: task_* + dimensions) against the judge labels.
python prompt/eval_prompt.py --models A,B --ref TEST_JUDGED.parquet --out R.json"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from metrics import metrics

ap = argparse.ArgumentParser(); ap.add_argument("--models", required=True); ap.add_argument("--ref", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); args = ap.parse_args()
ref = pd.read_parquet(args.ref); res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec); m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    labels = [m.config.id2label[i] for i in range(m.config.num_labels)]; P = np.zeros((len(ref), len(labels)))
    order = np.argsort(ref.text.str.len().values)
    with torch.no_grad():
        for s in range(0, len(ref), args.bs):
            b = order[s:s + args.bs]; e = tok(ref.text.iloc[b].tolist(), truncation=True, max_length=512, padding=True, return_tensors="pt")
            P[b] = torch.sigmoid(m(input_ids=e["input_ids"].cuda(), attention_mask=e["attention_mask"].cuda()).logits.float()).cpu().numpy()
    pred = pd.DataFrame(P, columns=labels); r = metrics(pred, ref, ref.lang.values)
    print(spec, {k: round(v, 4) for k, v in r.items() if isinstance(v, float)}, flush=True); res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
