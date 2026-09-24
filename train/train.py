"""Fine-tune an encoder as a binary prompt-injection classifier.

python train/train.py --base jhu-clsp/mmBERT-small --data $DATA --out $OUTPUT_DIR/model
Data: parquet with text,label (+ source, kind, lang); see data/build_v0.py.
"""
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, torch
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import roc_auc_score, f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=1024)
ap.add_argument("--tokens_per_batch", type=int, default=65536)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--labels", default="SAFE,INJECTION", help="names for label 0,1")
ap.add_argument("--drop_sources", default="", help="regex; training/val rows whose source matches are dropped")
ap.add_argument("--eval_every", type=int, default=500)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(
    args.base, num_labels=2, id2label=dict(enumerate(args.labels.split(","))), label2id={l: i for i, l in enumerate(args.labels.split(","))}
).cuda()

tr = pd.read_parquet(f"{args.data}/train.parquet")
va = pd.read_parquet(f"{args.data}/val.parquet")
if args.drop_sources:
    tr = tr[~tr.source.str.match(args.drop_sources)].reset_index(drop=True)
    va = va[~va.source.str.match(args.drop_sources)].reset_index(drop=True)
print("train", len(tr), "pos", tr.label.mean(), "val", len(va), flush=True)


def encode(df):
    if "text_pair" in df.columns:  # (context, response) pairs: truncate the context first
        enc = tok(df.text.tolist(), df.text_pair.tolist(), truncation="longest_first", max_length=args.max_len)
    else:
        enc = tok(df.text.tolist(), truncation=True, max_length=args.max_len)
    return [(ids, int(l)) for ids, l in zip(enc["input_ids"], df.label)]


t0 = time.time()
tr_enc, va_enc = encode(tr), encode(va)
print(f"tokenized in {time.time()-t0:.0f}s; mean len {np.mean([len(x[0]) for x in tr_enc]):.0f}", flush=True)


def batches(data, shuffle):
    """Length-bucketed batches under a token budget."""
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    # sort within chunks of 100 batches' worth for bucketing
    out, chunk = [], 4096
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s : s + chunk], key=lambda i: len(data[i][0]))
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
    pad = tok.pad_token_id
    ids = torch.full((len(b), L), pad, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for j, i in enumerate(b):
        x = data[i][0]
        ids[j, : len(x)] = torch.tensor(x); att[j, : len(x)] = 1
    y = torch.tensor([data[i][1] for i in b])
    return ids.cuda(non_blocking=True), att.cuda(non_blocking=True), y.cuda(non_blocking=True)


@torch.no_grad()
def evaluate():
    model.eval()
    probs = np.zeros(len(va_enc))
    for b in batches(va_enc, False):
        ids, att, _ = collate(va_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(input_ids=ids, attention_mask=att).logits.float()
        probs[b] = torch.softmax(lo, -1)[:, 1].cpu().numpy()
    model.train()
    y = va.label.values
    pred = probs > 0.5
    res = dict(auc=roc_auc_score(y, probs), f1=f1_score(y, pred), acc=(pred == y).mean(),
               fpr=pred[y == 0].mean(), fnr=(~pred[y == 1]).mean())
    per = {}
    for s in va.source.unique():
        m = (va.source == s).values
        per[s] = round(float((pred[m] == y[m]).mean()), 3)
    return res, per, probs


steps_per_epoch = len(batches(tr_enc, True))
total = int(steps_per_epoch * args.epochs)
no_decay = ["bias", "norm", "LayerNorm"]
groups = [
    {"params": [p for n, p in model.named_parameters() if not any(k in n for k in no_decay)], "weight_decay": args.wd},
    {"params": [p for n, p in model.named_parameters() if any(k in n for k in no_decay)], "weight_decay": 0.0},
]
opt = torch.optim.AdamW(groups, lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=True)
sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
print(f"steps/epoch {steps_per_epoch}, total {total}", flush=True)

step, best, log = 0, -1, []
model.train()
t0 = time.time()
while step < total:
    for b in batches(tr_enc, True):
        ids, att, y = collate(tr_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        loss = torch.nn.functional.cross_entropy(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total:
            res, per, _ = evaluate()
            log.append(dict(step=step, **res))
            print("EVAL", step, json.dumps({k: round(float(v), 4) for k, v in res.items()}), flush=True)
            if res["f1"] > best:
                best = res["f1"]
                model.save_pretrained(args.out); tok.save_pretrained(args.out)
                json.dump(dict(step=step, **{k: float(v) for k, v in res.items()}, per_source=per), open(f"{args.out}/val_metrics.json", "w"), indent=1)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log), open(f"{args.out}/train_log.json", "w"), indent=1, default=float)
print("done; best val f1", best, flush=True)
