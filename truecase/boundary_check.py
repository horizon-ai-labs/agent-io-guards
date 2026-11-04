"""Comma -> capital boundary errors of a joint truecase model, and a boundary-consistent decoding.
The mark of word i and the case of word i+1 are separate labels, so the model can write "hello, How are you". Where the argmax gives
(',' after word i, Cap on word i+1), compare three options with the marginal probabilities: keep (',' , Cap) * beta, ('.'/'?' , Cap),
(',' , lower). beta = 1 is the plain argmax. beta is chosen on held-out VALIDATION windows (val.parquet rows after the 6,000 used for
checkpoint selection), then the test sets are scored with every beta for the record.
python truecase/boundary_check.py OUT.json DATA_DIR EVAL_DIR MODEL [MODEL ...]"""
import json, os, sys
import numpy as np, pandas as pd, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "punct"))
from common import LABELS as PL, NOSPACE, lower_keep_len
from common_tc import unit_cases
from tc_metrics import chunks, score
from transformers import AutoModelForTokenClassification, AutoTokenizer

out_f, data, ev = sys.argv[1:4]; models = sys.argv[4:]
BETAS = [1.0, 0.8, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
COMMA, PER, Q = PL.index(","), PL.index("."), PL.index("?")

va = pd.read_parquet(f"{data}/val.parquet").sample(frac=1.0, random_state=0).iloc[6000:]
val_items = [dict(set="val", lang=l, text=t, ends=e.tolist(), gp=[PL[x] for x in lb], gc=unit_cases(t, e.tolist(), l in NOSPACE), nos=l in NOSPACE)
             for t, e, lb, l in zip(va.text, va.ends, va.labs, va.lang)]
test_items = [it for it in chunks(ev) if it["set"] in ("europarl", "ted", "flores")]
print("val", len(val_items), "test", len(test_items), flush=True)


def probs(tok, model, items):
    texts = [lower_keep_len(it["text"]) for it in items]; order = np.argsort([len(t) for t in texts]); res = [None] * len(items)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=512, padding=True, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping").tolist()
        with torch.no_grad():
            p = torch.softmax(model(**{k: v.cuda() for k, v in enc.items()}).logits.float(), -1).cpu().numpy()
        for j, i in enumerate(idx):
            at = {e: k for k, (b, e) in enumerate(offs[j]) if e > b}
            P = np.zeros((len(items[i]["ends"]), 18), np.float32); P[:, 0] = 1.0
            for u, e in enumerate(items[i]["ends"]):
                if e in at:
                    P[u] = p[j, at[e]]
            res[i] = P
    return res


def decode(P, beta):
    j = P.argmax(1); pm, pc = (j // 3).tolist(), (j % 3).tolist()
    if beta < 1:
        M, C = P.reshape(-1, 6, 3).sum(2), P.reshape(-1, 6, 3).sum(1)
        for i in range(len(pm) - 1):
            if pm[i] == COMMA and pc[i + 1] == 1:
                b = PER if M[i, PER] >= M[i, Q] else Q
                z, x, y = M[i, COMMA] * C[i + 1, 1] * beta, M[i, b] * C[i + 1, 1], M[i, COMMA] * C[i + 1, 0]
                if x >= z and x >= y:
                    pm[i] = b
                elif y > z:
                    pc[i + 1] = 0
    return [PL[m] for m in pm], pc


def events(items, PP):
    """raw argmax (',' , Cap) boundaries by gold: sentence boundary, comma + capital (name/I), comma/none + lower (wrong capital)."""
    n = dict(total=0, gold_sentence_break=0, gold_comma_cap=0, gold_lower=0, other=0, words=0)
    for it, P in zip(items, PP):
        pm, pc = decode(P, 1.0); n["words"] += len(pm)
        for i in range(len(pm) - 1):
            if pm[i] == "," and pc[i + 1] == 1:
                n["total"] += 1; g, gc = it["gp"][i], it["gc"][i + 1]
                k = "gold_sentence_break" if g in (".", "?") and gc == 1 else "gold_comma_cap" if gc == 1 else "gold_lower" if gc == 0 else "other"
                n[k] += 1
    return n


res = json.load(open(out_f)) if os.path.exists(out_f) else {}
for m in models:
    tok = AutoTokenizer.from_pretrained(m); model = AutoModelForTokenClassification.from_pretrained(m, dtype=torch.bfloat16).cuda().eval()
    PV, PT = probs(tok, model, val_items), probs(tok, model, test_items)
    r = dict(events_val=events(val_items, PV), events_test={s: events([it for it in test_items if it["set"] == s], [P for it, P in zip(test_items, PT) if it["set"] == s])
                                                          for s in ("europarl", "ted", "flores")}, val={}, test={})
    for b in BETAS:
        dv = [decode(P, b) for P in PV]; dt = [decode(P, b) for P in PT]
        r["val"][b] = score(val_items, [d[0] for d in dv], [d[1] for d in dv])["summary"]
        r["test"][b] = score(test_items, [d[0] for d in dt], [d[1] for d in dt])["summary"]
        v = r["val"][b]["val"]; print(m[-50:], "beta", b, "val", round(v["punct_f1"], 4), round(v["case_f1"], 4),
                                      {s: (round(x["punct_f1"], 4), round(x["case_f1"], 4)) for s, x in r["test"][b].items()}, flush=True)
    best = max(BETAS, key=lambda b: (r["val"][b]["val"]["punct_f1"] + r["val"][b]["val"]["case_f1"]) / 2); r["best_beta_val"] = best
    print(m[-50:], "events", r["events_val"], r["events_test"], "best beta (val)", best, flush=True)
    res[m] = r; json.dump(res, open(out_f, "w"), indent=1)
