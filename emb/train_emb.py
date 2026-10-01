"""Distil BAAI/bge-m3 dense embeddings into a smaller encoder (same 1024-d vector space).
python emb/train_emb.py --base jhu-clsp/mmBERT-small --teacher DIR[,DIR] --out OUT
Student = encoder + mean pooling + linear projection (hidden -> 1024, saved as OUT/projection.pt), L2-normalised.
Loss = (1 - cos(student, teacher)) + mse_w * ||student - teacher||^2 + sim_w * in-batch similarity-matrix MSE (keeps the
relative geometry). Val = 0.5% of texts: mean cosine to the teacher and top-1 in-batch retrieval agreement.
"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small"); ap.add_argument("--teacher", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=256); ap.add_argument("--bs", type=int, default=256); ap.add_argument("--lr", type=float, default=1e-4)
ap.add_argument("--epochs", type=float, default=2.0); ap.add_argument("--warmup", type=float, default=0.03)
ap.add_argument("--mse_w", type=float, default=1.0); ap.add_argument("--sim_w", type=float, default=1.0)
ap.add_argument("--seed", type=int, default=0); ap.add_argument("--eval_every", type=int, default=2000)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)
texts, embs = [], []
for d in args.teacher.split(","):
    texts += pd.read_parquet(f"{d}/texts.parquet").text.tolist(); embs.append(np.load(f"{d}/emb.npy", mmap_mode="r"))
E = np.concatenate([np.asarray(e) for e in embs]).astype(np.float16)
n = len(texts); rng = np.random.RandomState(0); isval = rng.rand(n) < 0.005
tr_idx, va_idx = np.where(~isval)[0], np.where(isval)[0][:4096]
print("texts", n, "train", len(tr_idx), "val", len(va_idx), flush=True)
tok = AutoTokenizer.from_pretrained(args.base); enc = AutoModel.from_pretrained(args.base).cuda()
proj = torch.nn.Linear(enc.config.hidden_size, 1024, bias=False).cuda()
params = list(enc.parameters()) + list(proj.parameters())


def embed(idx):
    e = tok([texts[i] for i in idx], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt").to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = enc(input_ids=e["input_ids"], attention_mask=e["attention_mask"]).last_hidden_state
    a = e["attention_mask"].unsqueeze(-1).float(); v = (h.float() * a).sum(1) / a.sum(1)
    return F.normalize(proj(v), dim=-1)


def loss_fn(s, t):
    cos = (s * t).sum(-1)
    return (1 - cos).mean() + args.mse_w * ((s - t) ** 2).sum(-1).mean() + args.sim_w * ((s @ s.T - t @ t.T) ** 2).mean() * 100


@torch.no_grad()
def evaluate():
    enc.eval(); S = []
    for s in range(0, len(va_idx), 256):
        S.append(embed(va_idx[s:s + 256]))
    enc.train(); S = torch.cat(S); T = torch.tensor(E[va_idx], dtype=torch.float32, device="cuda")
    top = ((S @ T.T).argmax(1) == torch.arange(len(S), device="cuda")).float().mean().item()
    return dict(cos=float((S * T).sum(-1).mean().item()), top1=top)


steps = int(len(tr_idx) / args.bs * args.epochs)
opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.01, betas=(0.9, 0.98), eps=1e-6, fused=True)
sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * steps), steps)
print("steps", steps, flush=True)
step, best, log, t0 = 0, -1, [], time.time()
while step < steps:
    perm = np.random.permutation(tr_idx)
    for s in range(0, len(perm) - args.bs + 1, args.bs):
        b = perm[s:s + args.bs]
        st = embed(b); tt = torch.tensor(E[b], dtype=torch.float32, device="cuda")
        loss = loss_fn(st, tt); loss.backward(); torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 200 == 0:
            print(f"step {step}/{steps} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == steps:
            r = evaluate(); log.append(dict(step=step, **r)); print("EVAL", step, json.dumps(r), flush=True)
            if r["cos"] > best:
                best = r["cos"]; enc.save_pretrained(args.out); tok.save_pretrained(args.out); torch.save(proj.weight.detach().T.contiguous().cpu(), f"{args.out}/projection.pt")
                json.dump(dict(step=step, **r), open(f"{args.out}/val_metrics.json", "w"), indent=1)
        if step >= steps:
            break
json.dump(dict(args=vars(args), log=log), open(f"{args.out}/train_log.json", "w"), indent=1)
print("done; best val cos", best)
