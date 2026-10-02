"""Evaluate token-classification NER models on ner/evals.   python ner/eval_ner.py --models A,B --evals ner/evals --out R.json
Words are given (is_split_into_words); each word gets the label of its first sub-token. Labels are mapped to IOB2 with
PER/ORG/LOC/MISC (other types, e.g. DATE, -> O; types the set does not annotate -> O). Metric: entity-level micro F1
(exact span and type), per set; family means _conll, _wikineural, _wikiann.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=32)
ap.add_argument("--mode", default="raw", choices=["raw", "words"],
                help="raw: words joined into text (no spaces for zh/ja/th), tokens mapped back by character offsets (how the pipeline "
                     "is used); words: is_split_into_words")
args = ap.parse_args()
EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}
TYPES = {"PER": "PER", "PERSON": "PER", "ORG": "ORG", "ORGANIZATION": "ORG", "LOC": "LOC", "LOCATION": "LOC", "GPE": "LOC", "MISC": "MISC"}


def norm_label(l):
    l = l.upper()
    if l == "O" or "-" not in l and l not in TYPES:
        return "O"
    p, t = (l.split("-", 1) if "-" in l else ("I", l))
    t = TYPES.get(t)
    return "O" if t is None else f"{'B' if p in ('B', 'S') else 'I'}-{t}"


def spans(tags):
    out, cur = set(), None
    for i, t in enumerate(list(tags) + ["O"]):
        if t == "O" or t.startswith("B-") or (cur and t[2:] != cur[1]):
            if cur:
                out.add((cur[0], i, cur[1])); cur = None
        if t != "O" and cur is None:
            cur = (i, t[2:])
    return out


res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec, add_prefix_space=True) if "roberta" in spec.lower() else AutoTokenizer.from_pretrained(spec)
    m = AutoModelForTokenClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    lab = [norm_label(m.config.id2label[i]) for i in range(m.config.num_labels)]
    print(spec, sorted(set(lab)), flush=True)
    r = {}
    for name, df in EV.items():
        allowed = set(df.classes.iloc[0]); tp = fp = fn = 0
        sents = [list(t) for t in df.tokens]; nospace = name.endswith(("_zh", "_ja", "_th"))
        for s in range(0, len(sents), args.bs):
            batch = sents[s:s + args.bs]
            if args.mode == "words":
                enc = tok(batch, is_split_into_words=True, truncation=True, max_length=512, padding=True, return_tensors="pt")
            else:
                texts, wspans = [], []
                for words in batch:
                    t, sp = "", []
                    for w in words:
                        if t and not nospace:
                            t += " "
                        sp.append((len(t), len(t) + len(w))); t += w
                    texts.append(t); wspans.append(sp)
                enc = tok(texts, truncation=True, max_length=512, padding=True, return_tensors="pt", return_offsets_mapping=True)
                offs = enc.pop("offset_mapping").numpy()
            with torch.no_grad():
                pred = m(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask", "token_type_ids")}).logits.argmax(-1).cpu().numpy()
            for b, words in enumerate(batch):
                wl = ["O"] * len(words); seen = set()
                if args.mode == "words":
                    pairs = [(j, w) for j, w in enumerate(enc.word_ids(b)) if w is not None]
                else:   # token j -> word whose character span contains the token start
                    starts = [a for a, _ in wspans[b]]; pairs = []
                    for j, (a, e) in enumerate(offs[b]):
                        while a < e and texts[b][a] == " ":   # SentencePiece offsets can include the leading space
                            a += 1
                        if e > a and enc["attention_mask"][b, j]:
                            k = int(np.searchsorted(starts, a, side="right")) - 1
                            if k >= 0 and a < wspans[b][k][1] + 1:
                                pairs.append((j, k))
                for j, w in pairs:
                    if w not in seen:
                        seen.add(w); l = lab[pred[b, j]]; wl[w] = l if l == "O" or l[2:] in allowed else "O"
                gold = spans(df.tags.iloc[s + b]); prd = spans(wl)
                tp += len(gold & prd); fp += len(prd - gold); fn += len(gold - prd)
        r[name] = dict(n=len(df), f1=float(2 * tp / (2 * tp + fp + fn)) if tp else 0.0, p=float(tp / max(1, tp + fp)), r=float(tp / max(1, tp + fn)))
    fam = lambda p: [v["f1"] for k, v in r.items() if k.startswith(p)]
    r["_conll"] = float(np.mean(fam("conll_"))); r["_wikineural"] = float(np.mean(fam("wikineural_"))); r["_wikiann"] = float(np.mean(fam("wikiann_")))
    print(spec, {k: round(v, 4) for k, v in r.items() if k.startswith("_")}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
