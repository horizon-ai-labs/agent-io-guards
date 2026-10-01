"""Fine-tune an encoder as a multi-label emotion classifier with the 28 GoEmotions labels (sigmoid, BCE).
python emotion/train_emo.py --base jhu-clsp/mmBERT-small --data $DATA --out $OUTPUT_DIR/model
Data: emotion/build_emo.py (train/val parquet: text, labels (list), lang, src_idx). Checkpoint selection on val (held-out
go_emotions train comments and their translations): macro-F1 over the 28 labels at threshold 0.5.
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=256)
ap.add_argument("--pos_weight", type=float, default=1.0, help="BCE weight of positive labels")
ap.add_argument("--tokens_per_batch", type=int, default=32768)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=3)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=1000)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GO
LABELS = GO
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                           label2id={l: i for i, l in enumerate(LABELS)},
                                                           problem_type="multi_label_classification").cuda()
tr = pd.read_parquet(f"{args.data}/train.parquet"); va = pd.read_parquet(f"{args.data}/val.parquet")
print("train", len(tr), "val", len(va), flush=True)


def encode(df):
    enc = tok(df.text.tolist(), truncation=True, max_length=args.max_len)["input_ids"]
    Y = np.array([[l in set(ls) for l in LABELS] for ls in df.labels], dtype=np.float32)
    return list(zip(enc, Y))


tr_enc, va_enc = encode(tr), encode(va)


def batches(data, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    out = []
    for s in range(0, len(idx), 4096):
        part = sorted(idx[s: s + 4096], key=lambda i: len(data[i][0]))
        cur, mx = [], 0
        for i in part:
            L = len(data[i][0])
            if cur and max(mx, L) * (len(cur) + 1) > args.tokens_per_batch:
                out.append(cur); cur, mx = [], 0
            cur.append(i); mx = max(mx, L)
        if cur:
            out.append(cur)
    if shuffle:
        random.shuffle(out)
    return out


def collate(data, b):
    L = max(len(data[i][0]) for i in b)
    ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long); att = torch.zeros((len(b), L), dtype=torch.long)
    for j, i in enumerate(b):
        x = data[i][0]; ids[j, :len(x)] = torch.tensor(x); att[j, :len(x)] = 1
    return ids.cuda(), att.cuda(), torch.tensor(np.stack([data[i][1] for i in b])).cuda()


@torch.no_grad()
def evaluate():
    model.eval(); P = np.zeros((len(va_enc), len(LABELS)))
    for b in batches(va_enc, False):
        ids, att, _ = collate(va_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            P[b] = torch.sigmoid(model(input_ids=ids, attention_mask=att).logits.float()).cpu().numpy()
    model.train()
    T = np.stack([x[1] for x in va_enc]).astype(bool); p = P >= 0.5
    f1 = []
    for c in range(len(LABELS)):
        tp = (p[:, c] & T[:, c]).sum(); f1.append(0.0 if tp == 0 else 2 * tp / (2 * tp + (p[:, c] & ~T[:, c]).sum() + (~p[:, c] & T[:, c]).sum()))
    return dict(macro_f1=float(np.mean(f1)), exact=float((p == T).all(1).mean()),
                bce=float(-(T * np.log(np.clip(P, 1e-9, 1)) + ~T * np.log(np.clip(1 - P, 1e-9, 1))).mean()))


steps = len(batches(tr_enc, True)); total = int(steps * args.epochs)
nd = ["bias", "norm", "LayerNorm"]
opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(k in n for k in nd)], "weight_decay": args.wd},
                         {"params": [p for n, p in model.named_parameters() if any(k in n for k in nd)], "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=True)
sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
print(f"steps/epoch {steps}, total {total}", flush=True)
step, best, log, t0 = 0, -1, [], time.time()
model.train()
while step < total:
    for b in batches(tr_enc, True):
        ids, att, t = collate(tr_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, t, pos_weight=torch.full((len(LABELS),), args.pos_weight, device=t.device))
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total:
            res = evaluate(); log.append(dict(step=step, **res)); print("EVAL", step, json.dumps(res), flush=True)
            if res["macro_f1"] > best:
                best = res["macro_f1"]; model.save_pretrained(args.out); tok.save_pretrained(args.out)
                json.dump(dict(step=step, **res), open(f"{args.out}/val_metrics.json", "w"), indent=1)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log), open(f"{args.out}/train_log.json", "w"), indent=1)
print("done; best val macro-F1", best)
