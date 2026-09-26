"""Per-category decision thresholds for a content-safety model: chosen on the v0 validation set (best F1 on unsafe items),
reported on the v0 test set.  python safety/cat_thresholds.py MODEL_DIR V0_DIR OUT.json"""
import json, sys
import numpy as np, pandas as pd, torch
from sklearn.metrics import f1_score, roc_auc_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer

md, v0, out = sys.argv[1:4]
tok = AutoTokenizer.from_pretrained(md)
model = AutoModelForSequenceClassification.from_pretrained(md, torch_dtype=torch.bfloat16).cuda().eval()
labels = [model.config.id2label[i] for i in range(model.config.num_labels)]


@torch.no_grad()
def probs(df):
    enc = [tok(t, p, truncation="longest_first", max_length=1024)["input_ids"] if p else tok(t, truncation=True, max_length=1024)["input_ids"]
           for t, p in zip(df.text, df.text_pair)]
    out = np.zeros((len(df), len(labels)))
    order = np.argsort([len(x) for x in enc])
    for s in range(0, len(df), 64):
        b = order[s: s + 64]; L = max(len(enc[i]) for i in b)
        ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
        for j, i in enumerate(b):
            ids[j, :len(enc[i])] = torch.tensor(enc[i]); att[j, :len(enc[i])] = 1
        out[b] = torch.sigmoid(model(input_ids=ids.cuda(), attention_mask=att.cuda()).logits.float()).cpu().numpy()
    return out


res = {}
va, te = [pd.read_parquet(f"{v0}/{s}.parquet").sample(n=8000, random_state=0) for s in ["val", "test"]]
pv, pt = probs(va), probs(te)
for k, c in enumerate(labels):
    if c == "unsafe":
        continue
    mv, mt = va.unsafe.values == 1, te.unsafe.values == 1
    yv = np.array([c in list(x) for x in va.cats.values[mv]]); yt = np.array([c in list(x) for x in te.cats.values[mt]])
    thr = max((f1_score(yv, pv[mv, k] > t, zero_division=0), t) for t in np.arange(0.05, 0.95, 0.05))[1]
    res[c] = dict(threshold=round(float(thr), 2), test_f1=round(float(f1_score(yt, pt[mt, k] > thr, zero_division=0)), 3),
                  test_f1_at_0_5=round(float(f1_score(yt, pt[mt, k] > 0.5, zero_division=0)), 3),
                  test_auc=round(float(roc_auc_score(yt, pt[mt, k])), 3), test_positives=int(yt.sum()))
    print(c, res[c], flush=True)
json.dump(dict(unsafe=0.5, categories={c: v["threshold"] for c, v in res.items()}, test=res,
               note="category thresholds chosen on the Nemotron-Safety-Guard-v3 validation split (best F1 on unsafe items); "
                    "test numbers on its test split. Apply categories only when unsafe >= its threshold."), open(out, "w"), indent=1)
