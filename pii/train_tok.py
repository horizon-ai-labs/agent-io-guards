"""Fine-tune an encoder for PII span detection (BIO token classification).

python pii/train_tok.py --base jhu-clsp/mmBERT-small --data $DATA --out $OUTPUT_DIR/model
Data: parquet with text, spans_json ([[start, end, LABEL], ...]); see pii/build_pii_v0.py.
Metrics: entity-level exact-span F1 (micro, per label) and character-level redaction recall/precision
(ignoring labels: what fraction of PII characters would be masked).
"""
import argparse, json, os, random, time
from collections import defaultdict
import numpy as np, pandas as pd, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer, get_cosine_schedule_with_warmup

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="jhu-clsp/mmBERT-small")
ap.add_argument("--data", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--max_len", type=int, default=512)
ap.add_argument("--stride", type=int, default=128)
ap.add_argument("--tokens_per_batch", type=int, default=65536)
ap.add_argument("--lr", type=float, default=5e-5)
ap.add_argument("--epochs", type=float, default=2)
ap.add_argument("--warmup", type=float, default=0.05)
ap.add_argument("--synth_repeat", type=int, default=1, help="repeat rows whose source starts with synthetic_ N times")
ap.add_argument("--drop_sources", default="", help="regex on source; matching train/val rows are dropped")
ap.add_argument("--max_train", type=int, default=0)
ap.add_argument("--eval_every", type=int, default=1000)
ap.add_argument("--seed", type=int, default=0)
args = ap.parse_args()
random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
os.makedirs(args.out, exist_ok=True)

tr = pd.read_parquet(f"{args.data}/train.parquet")
va = pd.read_parquet(f"{args.data}/val.parquet")
if args.drop_sources:
    tr = tr[~tr.source.str.match(args.drop_sources)]; va = va[~va.source.str.match(args.drop_sources)]
if args.synth_repeat > 1:
    syn = tr[tr.source.str.startswith("synthetic_")]
    tr = pd.concat([tr] + [syn] * (args.synth_repeat - 1)).sample(frac=1.0, random_state=0)
    print("synthetic rows", len(syn), "x", args.synth_repeat, flush=True)
if args.max_train:
    tr = tr.sample(min(len(tr), args.max_train), random_state=0)
LABELS = sorted({l for s in pd.concat([tr, va]).spans_json for _, _, l in json.loads(s)})
tags = ["O"] + [f"{p}-{l}" for l in LABELS for p in ("B", "I")]
t2i = {t: i for i, t in enumerate(tags)}
print("labels", LABELS, flush=True)

tok = AutoTokenizer.from_pretrained(args.base)
model = AutoModelForTokenClassification.from_pretrained(
    args.base, num_labels=len(tags), id2label=dict(enumerate(tags)), label2id=t2i).cuda()


def encode(df, with_labels=True):
    """-> list of (ids, labels, offsets, doc_index) windows."""
    enc = tok(df.text.tolist(), truncation=True, max_length=args.max_len, stride=args.stride,
              return_overflowing_tokens=True, return_offsets_mapping=True)
    out = []
    spans = [json.loads(s) for s in df.spans_json]
    for w, d in enumerate(enc["overflow_to_sample_mapping"]):
        ids, offs = enc["input_ids"][w], enc["offset_mapping"][w]
        lab = []
        sp = spans[d]
        for (s, e) in offs:
            if s == e:
                lab.append(-100); continue
            t = "O"
            for a, b, L in sp:
                if s < b and e > a:  # token overlaps span
                    t = ("B-" if s <= a else "I-") + L
                    break
            lab.append(t2i[t])
        # a span's first token may start before the span (leading space merged); ensure B at first overlap
        for i in range(1, len(lab)):
            if lab[i] > 0 and tags[lab[i]].startswith("I-") and (lab[i - 1] <= 0 or tags[lab[i - 1]][2:] != tags[lab[i]][2:]):
                lab[i] = t2i["B-" + tags[lab[i]][2:]]
        out.append((ids, lab, offs, d))
    return out


t0 = time.time()
tr_enc, va_enc = encode(tr), encode(va)
print(f"windows train {len(tr_enc)} val {len(va_enc)} in {time.time()-t0:.0f}s", flush=True)


def batches(data, shuffle):
    idx = list(range(len(data)))
    if shuffle:
        random.shuffle(idx)
    out = []
    for s in range(0, len(idx), 4096):
        part = sorted(idx[s:s + 4096], key=lambda i: len(data[i][0]))
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
    lab = torch.full((len(b), L), -100, dtype=torch.long)
    for j, i in enumerate(b):
        n = len(data[i][0])
        ids[j, :n] = torch.tensor(data[i][0]); att[j, :n] = 1; lab[j, :n] = torch.tensor(data[i][1])
    return ids.cuda(), att.cuda(), lab.cuda()


def decode_spans(pred, offs):
    """BIO tags -> character spans [(s, e, L)]; tolerant (I without B starts a span)."""
    spans, cur = [], None
    for p, (s, e) in zip(pred, offs):
        if s == e:
            continue
        t = tags[p]
        if t == "O":
            if cur: spans.append(cur); cur = None
            continue
        L = t[2:]
        if t.startswith("B-") or cur is None or cur[2] != L:
            if cur: spans.append(cur)
            cur = [s, e, L]
        else:
            cur[1] = e
    if cur:
        spans.append(cur)
    return spans


@torch.no_grad()
def predict(df, enc):
    model.eval()
    doc_spans = defaultdict(list)
    for b in batches(enc, False):
        ids, att, _ = collate(enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(input_ids=ids, attention_mask=att).logits
        pr = lo.argmax(-1).cpu().numpy()
        for j, i in enumerate(b):
            n = len(enc[i][0])
            doc_spans[enc[i][3]] += decode_spans(pr[j, :n], enc[i][2])
    model.train()
    out = []
    for d in range(len(df)):
        ss = sorted(set(tuple(x) for x in doc_spans[d]))
        # windows overlap: merge overlapping same-label spans
        merged = []
        for s, e, L in ss:
            if merged and merged[-1][2] == L and s <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], e)
            else:
                merged.append([s, e, L])
        out.append(merged)
    return out


def strip_span(text, s, e):
    while s < e and not text[s].isalnum(): s += 1
    while e > s and not text[e - 1].isalnum(): e -= 1
    return s, e


def metrics(df, preds):
    tp = fp = fn = 0; per = defaultdict(lambda: [0, 0, 0])
    ch_tp = ch_gold = ch_pred = 0
    for text, sj, pr in zip(df.text, df.spans_json, preds):
        gold = {(*strip_span(text, s, e), L) for s, e, L in json.loads(sj)}
        pred = {(*strip_span(text, s, e), L) for s, e, L in pr}
        for g in gold:
            if g in pred: tp += 1; per[g[2]][0] += 1
            else: fn += 1; per[g[2]][2] += 1
        for p in pred - gold:
            fp += 1; per[p[2]][1] += 1
        gm = np.zeros(len(text), bool); pm = np.zeros(len(text), bool)
        for s, e, _ in gold: gm[s:e] = True
        for s, e, _ in pred: pm[s:e] = True
        ws = np.array([not c.isspace() for c in text], bool)
        ch_tp += (gm & pm & ws).sum(); ch_gold += (gm & ws).sum(); ch_pred += (pm & ws).sum()
    f1 = lambda t, p, n: 2 * t / max(1, 2 * t + p + n)
    return dict(entity_f1=f1(tp, fp, fn), entity_p=tp / max(1, tp + fp), entity_r=tp / max(1, tp + fn),
                redact_recall=ch_tp / max(1, ch_gold), redact_precision=ch_tp / max(1, ch_pred),
                per_label={k: dict(f1=round(f1(*v), 4), n=v[0] + v[2]) for k, v in sorted(per.items())})


steps_per_epoch = len(batches(tr_enc, True))
total = int(steps_per_epoch * args.epochs)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.98), eps=1e-6, weight_decay=0.01, fused=True)
sch = get_cosine_schedule_with_warmup(opt, int(args.warmup * total), total)
print(f"steps/epoch {steps_per_epoch} total {total}", flush=True)
step, best, log = 0, -1, []
t0 = time.time()
model.train()
while step < total:
    for b in batches(tr_enc, True):
        ids, att, lab = collate(tr_enc, b)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            lo = model(input_ids=ids, attention_mask=att).logits.float()
        loss = torch.nn.functional.cross_entropy(lo.view(-1, lo.size(-1)), lab.view(-1), ignore_index=-100)
        loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step(); sch.step(); opt.zero_grad(set_to_none=True); step += 1
        if step % 100 == 0:
            print(f"step {step}/{total} loss {loss.item():.4f} {time.time()-t0:.0f}s", flush=True)
        if step % args.eval_every == 0 or step == total:
            m = metrics(va, predict(va, va_enc))
            log.append(dict(step=step, **{k: v for k, v in m.items() if k != "per_label"}))
            print("EVAL", step, json.dumps({k: round(v, 4) for k, v in m.items() if k != "per_label"}), flush=True)
            if m["entity_f1"] > best:
                best = m["entity_f1"]
                model.save_pretrained(args.out); tok.save_pretrained(args.out)
                json.dump(m, open(f"{args.out}/val_metrics.json", "w"), indent=1)
        if step >= total:
            break
json.dump(dict(args=vars(args), log=log), open(f"{args.out}/train_log.json", "w"), indent=1)

# final: evaluate best checkpoint on every eval set
model = AutoModelForTokenClassification.from_pretrained(args.out).cuda()
res = {}
import glob
for p in sorted(glob.glob(f"{args.data}/eval/*.parquet")):
    df = pd.read_parquet(p)
    m = metrics(df, predict(df, encode(df)))
    res[os.path.basename(p)[:-8]] = m
    print("TEST", os.path.basename(p)[:-8], json.dumps({k: round(v, 4) for k, v in m.items() if k != "per_label"}), flush=True)
json.dump(res, open(f"{os.path.dirname(args.out.rstrip('/'))}/eval_results.json", "w"), indent=1)
