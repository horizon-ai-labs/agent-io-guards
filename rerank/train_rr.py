"""Distil a cross-encoder reranker (1 logit) from teacher scores.
python rerank/train_rr.py --base jhu-clsp/mmBERT-small --data scored.parquet[,more] --passages passages.parquet --out DIR
Each step: B queries x G candidates (the source passage + G-1 sampled from the mined candidates). Loss = KL(softmax(teacher
log-odds / T) || softmax(student)) + bce_w * BCE(sigmoid(student), sigmoid(teacher log-odds)). Val = 1% of queries held
out: listwise KL and top-1 agreement with the teacher; checkpoint selection on val KL.
"""
import argparse, json, math, os, random, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification, AutoTokenizer, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small"); ap.add_argument("--data", required=True)
ap.add_argument("--passages", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=384); ap.add_argument("--group", type=int, default=16)
ap.add_argument("--queries_per_step", type=int, default=16); ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=1.0); ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--temp", type=float, default=1.0); ap.add_argument("--bce_w", type=float, default=0.5)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval_every", type=int, default=1000)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForSequenceClassification.from_pretrained(args.base, num_labels=1).cuda()
df = pd.concat([pd.read_parquet(p) for p in args.data.split(",")], ignore_index=True)
ptext = dict(pd.read_parquet(args.passages)[["pid", "text"]].values)
rng = np.random.RandomState(0); isval = rng.rand(len(df)) < 0.01
tr, va = df[~isval].reset_index(drop=True), df[isval].reset_index(drop=True)
print("train queries", len(tr), "val", len(va), flush=True)


def group(row, train):
    c, s = list(row.cands), list(row.scores)
    idx = [0] + (sorted(random.sample(range(1, len(c)), min(args.group - 1, len(c) - 1))) if train else list(range(1, min(args.group, len(c)))))
    return [(row.query, ptext[c[i]]) for i in idx], torch.tensor([s[i] for i in idx], dtype=torch.float32)


def forward(batch):
    pairs = [p for g, _ in batch for p in g]
    enc = tok([q for q, _ in pairs], [d for _, d in pairs], truncation="only_second", max_length=args.max_len, padding=True, return_tensors="pt")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        lo = model(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.float()[:, 0]
    return lo.view(len(batch), -1), torch.stack([t for _, t in batch]).cuda()


def loss_fn(s, t):
    kl = F.kl_div(F.log_softmax(s, -1), F.log_softmax(t / args.temp, -1), log_target=True, reduction="batchmean")
    return kl + args.bce_w * F.binary_cross_entropy_with_logits(s, torch.sigmoid(t)), kl


@torch.no_grad()
def evaluate():
    model.eval(); kls, top = [], []
    for s in range(0, len(va), args.queries_per_step):
        b = [group(r, False) for r in va.iloc[s:s + args.queries_per_step].itertuples()]
        st, tt = forward(b); _, kl = loss_fn(st, tt); kls.append(kl.item()); top += (st.argmax(1) == tt.argmax(1)).tolist()
    model.train()
    return dict(kl=float(np.mean(kls)), top1_agree=float(np.mean(top)))


steps_ep = len(tr) // args.queries_per_step; total = int(steps_ep * args.epochs)
nd = ["bias", "norm"]
opt = torch.optim.AdamW([{"params": [p for n, p in model.named_parameters() if not any(k in n for k in nd)], "weight_decay": 0.01},
                         {"params": [p for n, p in model.named_parameters() if any(k in n for k in nd)], "weight_decay": 0.0}],
                        lr=args.lr, betas=(0.9, 0.98), eps=1e-6, fused=True)
sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
print("steps", total, flush=True)
step, best, log, t0 = 0, 1e9, [], time.time()
model.train()
while step < total:
    order = np.random.permutation(len(tr))
    for s in range(0, len(order) - args.queries_per_step + 1, args.queries_per_step):
        b = [group(tr.iloc[i], True) for i in order[s:s + args.queries_per_step]]
        st, tt = forward(b); loss, kl = loss_fn(st, tt)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} kl {kl.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total:
            r = evaluate(); log.append(dict(step=step, **r)); print("EVAL", step, json.dumps(r), flush=True)
            if r["kl"] < best:
                best = r["kl"]; model.save_pretrained(args.out); tok.save_pretrained(args.out)
                json.dump(dict(step=step, **r), open(f"{args.out}/val_metrics.json", "w"), indent=1)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log), open(f"{args.out}/train_log.json", "w"), indent=1)
print("done; best val KL", best)
