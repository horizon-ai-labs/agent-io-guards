"""Train a multilingual LLMLingua-2-style prompt compressor: token classification, label 1 = keep, 0 = drop (as LLMLingua-2), every token of
a unit carries the unit's label (compress/align.py units and teacher labels). Checkpoint selection on validation: per window, keep the
k highest-scoring units (unit score = mean token P(keep), k = the teacher's kept count) and measure the overlap with the teacher's
kept units (top-k F1; precision = recall here), averaged over windows.
python compress/train_comp.py --base jhu-clsp/mmBERT-small --train A.parquet[,B...] --val V.parquet --out DIR"""
import argparse, json, os, random, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
ap.add_argument("--train", required=True)
ap.add_argument("--val", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--tokens_per_batch", type=int, default=65536)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--max_val", type=int, default=4000)
ap.add_argument("--evals", type=int, default=8)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForTokenClassification.from_pretrained(args.base, num_labels=2, id2label={0: "drop", 1: "keep"},
                                                        label2id={"drop": 0, "keep": 1}).cuda()


def encode(df):
    out = []
    for s0 in range(0, len(df), 20000):
        part = df.iloc[s0:s0 + 20000]
        enc = tok(part.text.tolist(), truncation=True, max_length=args.max_len, return_offsets_mapping=True)
        for ids, offs, st, en, kp in zip(enc["input_ids"], enc["offset_mapping"], part.starts, part.ends, part.keep):
            st = np.asarray(st); en = np.asarray(en); kp = np.asarray(kp); lab, unit = [], []
            for a, b in offs:
                k = int(np.searchsorted(st, a, side="right")) - 1
                ok = b > a and k >= 0 and a < en[k]
                lab.append(int(kp[k]) if ok else -100); unit.append(k if ok else -1)
            out.append((ids, lab, unit, int(kp[:max([u for u in unit if u >= 0], default=-1) + 1].sum())))
    return out


tr = pd.concat([pd.read_parquet(f) for f in args.train.split(",")]).sample(frac=1.0, random_state=args.seed)
va = pd.read_parquet(args.val).sample(frac=1.0, random_state=0).head(args.max_val)
t0 = time.time(); tr_enc, va_enc = encode(tr), encode(va)
print(f"train {len(tr_enc)} val {len(va_enc)} encoded in {time.time() - t0:.0f}s", flush=True)


def batches(data, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    out = []
    for s in range(0, len(idx), 8192):
        part = sorted(idx[s:s + 8192], key=lambda i: len(data[i][0])); cur, mx = [], 0
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
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    for j, i in enumerate(b):
        n = len(data[i][0]); ids[j, :n] = torch.tensor(data[i][0]); att[j, :n] = 1; lab[j, :n] = torch.tensor(data[i][1])
    return ids.cuda(), att.cuda(), lab.cuda()


@torch.no_grad()
def evaluate(data):
    model.eval(); f1s, acc_n, acc_d = [], 0, 0
    for b in batches(data, False):
        ids, att, lab = collate(data, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = torch.softmax(model(input_ids=ids, attention_mask=att).logits.float(), -1)[:, :, 1].cpu().numpy()
        labc = lab.cpu().numpy(); m = labc >= 0
        acc_n += int(((p >= .5) == (labc == 1))[m].sum()); acc_d += int(m.sum())
        for j, i in enumerate(b):
            _, tl, unit, k = data[i]; sc, gold = {}, {}
            for t, (u, l) in enumerate(zip(unit, tl)):
                if u >= 0:
                    sc.setdefault(u, []).append(p[j, t]); gold[u] = l
            if k <= 0 or not sc:
                continue
            us = sorted(sc, key=lambda u: -np.mean(sc[u]))[:k]
            f1s.append(sum(gold[u] for u in us) / k)
    model.train()
    return dict(topk_f1=round(float(np.mean(f1s)), 4), token_acc=round(acc_n / max(1, acc_d), 4))


nb = len(batches(tr_enc, True)); total = int(nb * args.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01, fused=True)
sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
every = max(1, total // args.evals)
print(f"steps/epoch {nb} total {total} eval every {every}", flush=True)
step, best, log, t0 = 0, -1, [], time.time()
model.train()
while step < total:
    for b in batches(tr_enc, True):
        ids, att, lab = collate(tr_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=att).logits.float().view(-1, 2), lab.view(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 200 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % every == 0 or step == total:
            v = evaluate(va_enc); log.append(dict(step=step, **v)); print(json.dumps(log[-1]), flush=True)
            if v["topk_f1"] > best:
                best = v["topk_f1"]; model.save_pretrained(args.out); tok.save_pretrained(args.out)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log, best=best), open(f"{args.out}/train_log.json", "w"), indent=1)
print("best", best, flush=True)
