"""Fine-tune an encoder for punctuation restoration (token classification, labels of punct/common.py).
python punct/train_punct.py --base jhu-clsp/mmBERT-small --data DATA_DIR --out OUT_DIR/model
Each word's label sits on its last token; other tokens are ignored. A fraction of the inputs is lowercased (ASR-style).
Checkpoint selection: punctuation F1 on the validation windows (cased and lowercased mean).
"""
import argparse, json, os, random, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_cosine_schedule_with_warmup
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, L2I, lower_keep_len

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--tokens_per_batch", type=int, default=65536)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=1)
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--lower", type=float, default=0.4)
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--max_val", type=int, default=6000)
ap.add_argument("--evals", type=int, default=8, help="validation checks over the run")
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

tr = pd.read_parquet(f"{args.data}/train.parquet")
va = pd.read_parquet(f"{args.data}/val.parquet").sample(frac=1.0, random_state=0).head(args.max_val)
if args.max_train:
    tr = tr.sample(min(len(tr), args.max_train), random_state=args.seed)
tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForTokenClassification.from_pretrained(args.base, num_labels=len(LABELS), id2label=dict(enumerate(LABELS)),
                                                        label2id=L2I).cuda()


def encode(df, lower):
    texts = [lower_keep_len(t) if lw else t for t, lw in zip(df.text, lower)]
    out = []
    for s in range(0, len(texts), 20000):
        enc = tok(texts[s:s + 20000], truncation=True, max_length=args.max_len, return_offsets_mapping=True)
        for ids, offs, ends, labs in zip(enc["input_ids"], enc["offset_mapping"], df.ends.iloc[s:s + 20000], df.labs.iloc[s:s + 20000]):
            at = dict(zip(ends.tolist(), labs.tolist()))
            out.append((ids, [at[e] if e > b and e in at else -100 for b, e in offs]))
    return out


t0 = time.time()
rng = np.random.RandomState(args.seed)
tr_enc = encode(tr, rng.rand(len(tr)) < args.lower)
va_c, va_l = encode(va, np.zeros(len(va), bool)), encode(va, np.ones(len(va), bool))
print(f"train {len(tr_enc)} val {len(va_c)} encoded in {time.time() - t0:.0f}s", flush=True)


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
    model.eval(); conf = np.zeros((len(LABELS), len(LABELS)), np.int64)
    for b in batches(data, False):
        ids, att, lab = collate(data, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            pr = model(input_ids=ids, attention_mask=att).logits.argmax(-1)
        m = lab >= 0
        np.add.at(conf, (lab[m].cpu().numpy(), pr[m].cpu().numpy()), 1)
    model.train()
    tp = sum(conf[i, i] for i in range(1, 6)); fp = conf[:, 1:].sum() - tp; fn = conf[1:, :].sum() - tp
    f1c = {LABELS[i]: round(2 * conf[i, i] / max(1, conf[i, :].sum() + conf[:, i].sum()), 4) for i in range(1, 6)}
    return dict(punct_f1=round(2 * tp / max(1, 2 * tp + fp + fn), 4), **f1c)


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
            loss = torch.nn.functional.cross_entropy(model(input_ids=ids, attention_mask=att).logits.float().view(-1, len(LABELS)),
                                                     lab.view(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); sch.step(); opt.zero_grad(set_to_none=True)
        step += 1
        if step % 200 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} {time.time() - t0:.0f}s", flush=True)
        if step % every == 0 or step == total:
            c, l = evaluate(va_c), evaluate(va_l); score = (c["punct_f1"] + l["punct_f1"]) / 2
            log.append(dict(step=step, cased=c, lower=l, score=score)); print(json.dumps(log[-1]), flush=True)
            if score > best:
                best = score; model.save_pretrained(args.out); tok.save_pretrained(args.out)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log, best=best), open(f"{args.out}/train_log.json", "w"), indent=1)
print("best", best, flush=True)
