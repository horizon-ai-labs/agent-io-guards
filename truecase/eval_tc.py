"""Evaluate punctuation + truecasing on the punctuation eval sets (punct/build_evals.py), always from LOWERCASED, mark-free input.
python truecase/eval_tc.py OUT.json EVAL_DIR SPEC [SPEC ...]
SPEC = joint:<model>  (our punctuation+truecasing model, labels '<mark>|<case>')
     | rule:<model>   (a punctuation-only model, labels '0 . , ? - :'; casing by rule: capitalise the first word and every word after a
                       predicted '.' or '?', and English 'i')
     | preds:<file.json>  (per-chunk predictions from another system, e.g. truecase/run_punctuators.py)
Metrics per set and language: punctuation F1 (as punct/eval_punct.py, lower mode), casing accuracy (all words) and casing F1 over the
upper-case classes (Cap, UPPER; a word counts only with the exact class); summary means over languages with cased scripts."""
import json, os, sys, time
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "punct"))
from common import LABELS as PL, NOSPACE, units, lower_keep_len  # noqa
from common_tc import CASES
from tc_metrics import chunks, score, unit_labels

out_f, ev = sys.argv[1:3]; specs = sys.argv[3:]


def run_model(mname, items):
    import torch
    from transformers import AutoModelForTokenClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(mname); model = AutoModelForTokenClassification.from_pretrained(mname, torch_dtype=torch.bfloat16).cuda().eval()
    id2 = [model.config.id2label[i] for i in range(model.config.num_labels)]; texts = [lower_keep_len(it["text"]) for it in items]
    order = np.argsort([len(t) for t in texts]); preds = [None] * len(items)
    for s in range(0, len(order), 64):
        idx = order[s:s + 64]
        enc = tok([texts[i] for i in idx], truncation=True, max_length=512, padding=True, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping").tolist()
        with torch.no_grad():
            pr = model(**{k: v.cuda() for k, v in enc.items()}).logits.argmax(-1).cpu().numpy()
        for j, i in enumerate(idx):
            preds[i] = unit_labels(offs[j], [id2[x] for x in pr[j].tolist()], items[i]["ends"], id2[0])
    return preds, texts


def rule_case(it, pp, text):
    out, cap = [], True; prev = 0
    for e, p in zip(it["ends"], pp):
        w = text[(e - 1 if it["nos"] else text.rfind(" ", 0, e) + 1):e]
        out.append(1 if (cap and any(ch.lower() != ch.upper() for ch in w)) or (it["lang"] == "eng_Latn" and w == "i") else 0)
        cap = p in (".", "?") if any(ch.lower() != ch.upper() for ch in w) or p in (".", "?") else cap
    return out


items = chunks(ev); print("chunks", len(items), flush=True)
res = json.load(open(out_f)) if os.path.exists(out_f) else {}
for spec in specs:
    kind, _, m = spec.partition(":"); t0 = time.time()
    if kind == "preds":
        P = json.load(open(m)); pp = [x["p"] for x in P]; pc = [x["c"] for x in P]
        keep = [i for i, x in enumerate(P) if x is not None and len(x["p"]) == len(items[i]["gp"])]
        r = score([items[i] for i in keep], [pp[i] for i in keep], [pc[i] for i in keep]); r["aligned_chunks"] = len(keep)
    else:
        preds, texts = run_model(m, items)
        if kind == "joint":
            pp = [[x.split("|")[0] if "|" in x else "0" for x in p] for p in preds]
            pc = [[CASES.index(x.split("|")[1]) if "|" in x else 0 for x in p] for p in preds]
        else:
            pp = [[x if x in PL else "0" for x in p] for p in preds]; pc = [rule_case(it, p, t) for it, p, t in zip(items, pp, texts)]
        r = score(items, pp, pc)
    r["seconds"] = round(time.time() - t0); res[spec] = r
    print(spec, {s: {k: round(v, 4) if isinstance(v, float) else v for k, v in d.items()} for s, d in r["summary"].items()}, flush=True)
    json.dump(res, open(out_f, "w"), indent=1)
