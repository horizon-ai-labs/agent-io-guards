"""Add teacher soft labels (P(positive class) from a trained classifier) to a training set (pairs or single texts).
python zeroshot/score_soft.py TEACHER_DIR DATA_DIR OUT_DIR  -> OUT_DIR/{train,val}.parquet with a 'soft' column"""
import os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
teacher, data, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(teacher)
m = AutoModelForSequenceClassification.from_pretrained(teacher, torch_dtype=torch.bfloat16).cuda().eval()
POS = os.environ.get("POS_LABEL", "entail")   # label name prefix of the positive class (e.g. INJECTION)
ei = [int(k) for k, v in m.config.id2label.items() if v.lower().startswith(POS.lower())][0]
for split in ["train", "val"]:
    df = pd.read_parquet(f"{data}/{split}.parquet")
    pair = "text_pair" in df.columns
    order = np.argsort((df.text.str.len() + (df.text_pair.str.len() if pair else 0)).values)
    soft = np.zeros(len(df), dtype=np.float32); t0 = time.time()
    bs = 256
    with torch.no_grad():
        for b in range(0, len(order), bs):
            idx = order[b:b + bs]
            if pair:
                enc = tok(df.text.values[idx].tolist(), df.text_pair.values[idx].tolist(), truncation="only_first", max_length=1024,
                          padding=True, return_tensors="pt").to("cuda")
            else:
                enc = tok(df.text.values[idx].tolist(), truncation=True, max_length=1024, padding=True, return_tensors="pt").to("cuda")
            enc.pop("token_type_ids", None)
            soft[idx] = torch.softmax(m(**enc).logits.float(), -1)[:, ei].cpu().numpy()
            if b % (bs * 500) == 0:
                print(split, b, len(df), f"{time.time() - t0:.0f}s", flush=True)
    df["soft"] = soft
    df.to_parquet(f"{out}/{split}.parquet")
    agree = ((soft > 0.5) == (df.label.values == 1)).mean()
    print(split, "done", len(df), "teacher agrees with hard label", round(float(agree), 4), flush=True)
