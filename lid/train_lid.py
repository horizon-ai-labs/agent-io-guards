"""Fine-tune an encoder as a language identifier (single-label, one class per FLORES-200 language code).
python lid/train_lid.py --base jhu-clsp/mmBERT-small --data $DATA --out $OUTPUT_DIR/model
Data: lid/prep_lid.py (train/val parquet: text, label, kind; labels.json).
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup
from sklearn.metrics import f1_score

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=128)
ap.add_argument("--tokens_per_batch", type=int, default=65536)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--warmup", type=float, default=0.06)
ap.add_argument("--wd", type=float, default=0.01)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=1000)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

LABELS = json.load(open(f"{args.data}/labels.json"))
L2I = {l: i for i, l in enumerate(LABELS)}
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                           label2id=L2I).cuda()
tr = pd.read_parquet(f"{args.data}/train.parquet"); va = pd.read_parquet(f"{args.data}/val.parquet")
print("train", len(tr), "val", len(va), "labels", len(LABELS), flush=True)


def encode(df):
    enc = tok(df.text.tolist(), truncation=True, max_length=args.max_len)["input_ids"]
    return [(ids, L2I[l]) for ids, l in zip(enc, df.label)]


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
    return ids.cuda(), att.cuda(), torch.tensor([data[i][1] for i in b]).cuda()


@torch.no_grad()
def evaluate():
    model.eval(); pred = np.zeros(len(va_enc), dtype=int)
    for b in batches(va_enc, False):
        ids, att, _ = collate(va_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pred[b] = model(input_ids=ids, attention_mask=att).logits.argmax(-1).cpu().numpy()
    model.train()
    y = np.array([x[1] for x in va_enc])
    res = dict(acc=float((pred == y).mean()), macro_f1=float(f1_score(y, pred, average="macro")))
    for k in ["long", "short"]:
        m = (va.kind == k).values
        res[f"acc_{k}"] = float((pred[m] == y[m]).mean())
    return res


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
        ids, att, y = collate(tr_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        loss = torch.nn.functional.cross_entropy(logits, y)
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
print("done; best val macro F1", best)
