"""Fine-tune an encoder as a multi-label content-safety classifier: label 0 = "unsafe", then harm categories.

python safety/train_safety.py --base jhu-clsp/mmBERT-small --data $DATA --out $OUTPUT_DIR/model
Data: safety/build_safety_v0.py (text, text_pair ("" = prompt only), unsafe, cats, lang, kind).
Sigmoid outputs (problem_type=multi_label_classification), so the transformers text-classification pipeline returns
independent scores with top_k=None. Loss = BCE(unsafe) + cat_weight * mean BCE(categories).
Optional columns: soft (teacher P(unsafe), NaN = use the hard label), has_cats (False = no category labels: category
loss masked for that row).
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
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
ap.add_argument("--cat_weight", type=float, default=1.0)
ap.add_argument("--eval_every", type=int, default=500)
ap.add_argument("--save_last", action="store_true", help="keep the final checkpoint instead of the best val F1")
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

CATS = json.load(open(f"{args.data}/stats.json"))["categories"]
LABELS = ["unsafe"] + CATS
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(
    args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)), label2id={l: i for i, l in enumerate(LABELS)},
    problem_type="multi_label_classification").cuda()

tr = pd.read_parquet(f"{args.data}/train.parquet")
va = pd.read_parquet(f"{args.data}/val.parquet")
print("train", len(tr), "unsafe", round(tr.unsafe.mean(), 3), "val", len(va), flush=True)


def encode(df):
    out = []
    soft = df.soft.values if "soft" in df.columns else np.full(len(df), np.nan)
    hc = df.has_cats.values if "has_cats" in df.columns else np.ones(len(df), dtype=bool)
    for t, p, u, c, sv, h in zip(df.text, df.text_pair, df.unsafe, df.cats, soft, hc):
        ids = tok(t, p, truncation="longest_first", max_length=args.max_len)["input_ids"] if p else \
            tok(t, truncation=True, max_length=args.max_len)["input_ids"]
        y = np.zeros(len(LABELS), dtype=np.float32); y[0] = u if np.isnan(sv) else sv
        for x in c:
            y[LABELS.index(x)] = 1
        out.append((ids, y, float(bool(h))))
    return out


t0 = time.time()
tr_enc, va_enc = encode(tr), encode(va)
print(f"tokenized in {time.time()-t0:.0f}s; mean len {np.mean([len(x[0]) for x in tr_enc]):.0f}", flush=True)


def batches(data, shuffle):
    """Length-bucketed batches under a token budget."""
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    out, chunk = [], 4096
    for s in range(0, len(idx), chunk):
        part = sorted(idx[s: s + chunk], key=lambda i: len(data[i][0]))
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
    ids = torch.full((len(b), L), tok.pad_token_id, dtype=torch.long)
    att = torch.zeros((len(b), L), dtype=torch.long)
    for j, i in enumerate(b):
        x = data[i][0]
        ids[j, : len(x)] = torch.tensor(x); att[j, : len(x)] = 1
    y = torch.tensor(np.stack([data[i][1] for i in b]))
    m = torch.tensor([data[i][2] for i in b], dtype=torch.float)
    return ids.cuda(non_blocking=True), att.cuda(non_blocking=True), y.cuda(non_blocking=True), m.cuda(non_blocking=True)


def loss_fn(logits, y, m):
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    cat = bce(logits[:, 1:], y[:, 1:], reduction="none").mean(-1)
    return bce(logits[:, 0], y[:, 0]) + args.cat_weight * (cat * m).sum() / m.sum().clamp(min=1)


@torch.no_grad()
def evaluate():
    model.eval()
    probs = np.zeros((len(va_enc), len(LABELS)))
    for b in batches(va_enc, False):
        ids, att, _, _ = collate(va_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            probs[b] = torch.sigmoid(model(input_ids=ids, attention_mask=att).logits.float()).cpu().numpy()
    model.train()
    Y = np.stack([x[1] for x in va_enc]); y = (Y[:, 0] > 0.5).astype(int); pred = probs[:, 0] > 0.5
    res = dict(auc=roc_auc_score(y, probs[:, 0]), f1=f1_score(y, pred), acc=(pred == y).mean(),
               fpr=pred[y == 0].mean(), fnr=(~pred[y == 1]).mean())
    m = y == 1   # category quality on unsafe items
    res["cat_macro_f1"] = np.mean([f1_score(Y[m, k], probs[m, k] > 0.5, zero_division=0) for k in range(1, len(LABELS)) if Y[m, k].sum() > 0])
    per = {}
    for col in ["kind", "lang"]:
        for s in va[col].unique():
            mm = (va[col] == s).values
            per[f"{col}={s}"] = round(float(f1_score(y[mm], pred[mm], zero_division=0)), 3)
    return res, per


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
        ids, att, y, m = collate(tr_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = model(input_ids=ids, attention_mask=att).logits.float()
        loss = loss_fn(logits, y, m)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 50 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} lr {sch.get_last_lr()[0]:.2e} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total:
            res, per = evaluate()
            log.append(dict(step=step, **res))
            print("EVAL", step, json.dumps({k: round(float(v), 4) for k, v in res.items()}), json.dumps(per), flush=True)
            if (res["f1"] > best and not args.save_last) or (args.save_last and step == total):
                best = max(best, res["f1"])
                model.save_pretrained(args.out); tok.save_pretrained(args.out)
                json.dump(dict(step=step, **{k: float(v) for k, v in res.items()}, per=per), open(f"{args.out}/val_metrics.json", "w"), indent=1)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log), open(f"{args.out}/train_log.json", "w"), indent=1, default=float)
print("done; best val f1", best, flush=True)
