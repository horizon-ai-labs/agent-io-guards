"""Label-agnostic PII redaction benchmark for token-classification models (ours and competitors).

python pii/evaluate_pii.py --models a,b --out results.json [--ours_data <pii data dir with eval/>]

For each benchmark we have gold character spans marked must-redact (and optionally may-redact).
Metrics (all ignore entity types, because every model has its own label set):
  redact_recall     share of must-redact characters (non-space) that the model masks
  redact_precision  share of masked characters that fall in any gold span (must or may)
  entity_recall     share of must-redact spans that are >= 90% covered
Models run through transformers' token-classification pipeline (aggregation="simple", stride windows).
"""
import argparse, json, os, re, time
from collections import defaultdict
import numpy as np, torch
from datasets import load_dataset
from transformers import pipeline, AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--ours_data", default="")
ap.add_argument("--max_docs", type=int, default=600)
ap.add_argument("--sets", default="", help="comma list of benchmark names to run (default all)")
ap.add_argument("--save_preds", action="store_true")
args = ap.parse_args()


def bench():
    B = {}
    # RedactionBench: realistic documents; spans 'mandatory' (must) vs 'contextual' (may)
    rows = []
    for r in load_dataset("RedactionBench/RedactionBench", split="test"):
        must = [(s["start"], s["end"]) for s in r["spans"] if s["label"] == "mandatory"]
        may = [(s["start"], s["end"]) for s in r["spans"] if s["label"] != "mandatory"]
        rows.append((r["raw_text"], must, may))
    B["redactionbench"] = rows
    # TonicAI Privacy-Bench: emails; all annotated spans are must
    rows = []
    for cfg in ["aaron_pfizer", "camille_nike", "elena_stripe", "hannah_salesforce", "malik_spotify", "marcus_boeing"]:
        for r in load_dataset("TonicAI/Privacy-Bench", "ground_truth_spans", split=cfg):
            rows.append((r["text"], [(s["start"], s["end"]) for s in r["ground_truth_spans"]], []))
    B["privacy_bench"] = rows
    # TAB (ECHR court cases, test split, first annotator per doc): DIRECT identifiers must, QUASI may
    rows, seen = [], set()
    for r in load_dataset("ildpil/text-anonymization-benchmark", split="test"):
        if r["doc_id"] in seen:
            continue
        seen.add(r["doc_id"])
        must = [(m["start_offset"], m["end_offset"]) for m in r["entity_mentions"] if m["identifier_type"] == "DIRECT"]
        may = [(m["start_offset"], m["end_offset"]) for m in r["entity_mentions"] if m["identifier_type"] == "QUASI"]
        rows.append((r["text"], must, may))
    B["tab_echr"] = rows
    # Russian PII benchmark (tokens + BIO tags -> char spans)
    rows = []
    for r in load_dataset("redmadrobot-rnd/pii_benchmark", split="test"):
        text = r["text"]; toks = json.loads(r["tokens"]); tags = json.loads(r["ner_tags"])
        pos, spans, cur = 0, [], None
        for t, g in zip(toks, tags):
            i = text.find(t, pos)
            if i < 0:
                continue
            pos = i + len(t)
            if g == "O":
                if cur: spans.append(tuple(cur)); cur = None
            elif g.startswith("B-") or cur is None:
                if cur: spans.append(tuple(cur))
                cur = [i, pos]
            else:
                cur[1] = pos
        if cur: spans.append(tuple(cur))
        rows.append((text, spans, []))
    B["ru_pii_benchmark"] = rows
    # our own held-out sets (in-distribution for our model)
    if args.ours_data:
        import pandas as pd, glob
        for p in sorted(glob.glob(f"{args.ours_data}/eval/*.parquet")):
            df = pd.read_parquet(p).sample(frac=1.0, random_state=0).head(args.max_docs)
            B["ours_" + os.path.basename(p)[:-8]] = [(t, [(s, e) for s, e, _ in json.loads(sj)], []) for t, sj in zip(df.text, df.spans_json)]
    if args.sets:
        B = {k: v for k, v in B.items() if k in args.sets.split(",")}
    for k in B:
        B[k] = B[k][: args.max_docs]
        print("bench", k, len(B[k]), "docs", sum(len(x[1]) for x in B[k]), "must spans", flush=True)
    return B


def score(text, must, may, pred):
    n = len(text)
    ws = np.array([not c.isspace() for c in text], bool)
    gm = np.zeros(n, bool); ga = np.zeros(n, bool); pm = np.zeros(n, bool)
    for s, e in must: gm[s:e] = True; ga[s:e] = True
    for s, e in may: ga[s:e] = True
    for s, e in pred: pm[max(0, s):min(n, e)] = True
    cov = [((pm[s:e] & ws[s:e]).sum() / max(1, ws[s:e].sum())) >= 0.9 for s, e in must]
    return (gm & pm & ws).sum(), (gm & ws).sum(), (pm & ga & ws).sum(), (pm & ws).sum(), sum(cov), len(cov)


B = bench()
results = json.load(open(args.out)) if os.path.exists(args.out) else {}
for name in args.models.split(","):
    t0 = time.time()
    try:
        tok = AutoTokenizer.from_pretrained(name, trust_remote_code=True)
        ml = tok.model_max_length if tok.model_max_length < 100000 else 512
        nlp = pipeline("token-classification", model=name, tokenizer=tok, aggregation_strategy="simple", device=0,
                       trust_remote_code=True, stride=min(128, ml // 4), dtype=torch.bfloat16)
        nlp.tokenizer.model_max_length = min(ml, 2048)
    except Exception as e:
        print("LOAD FAIL", name, repr(e)[:400], flush=True)
        results[name] = {"error": repr(e)[:400]}
        continue
    res = {}
    for bname, rows in B.items():
        acc = np.zeros(6); saved = []
        for text, must, may in rows:
            try:
                ents = nlp(text)
            except Exception as e:
                print("predict fail", name, bname, repr(e)[:200], flush=True); ents = []
            pred = [(int(x["start"]), int(x["end"])) for x in ents if x.get("start") is not None
                    and str(x.get("entity_group", x.get("entity", ""))).upper() not in ("O", "")]
            acc += score(text, must, may, pred)
            if args.save_preds:
                saved.append(dict(text=text, must=must, may=may, pred=pred,
                                  ents=[(int(x["start"]), int(x["end"]), str(x.get("entity_group"))) for x in ents]))
        r = dict(redact_recall=acc[0] / max(1, acc[1]), redact_precision=acc[2] / max(1, acc[3]), entity_recall=acc[4] / max(1, acc[5]))
        r = {k: float(v) for k, v in r.items()}
        r["redact_f1"] = 2 * r["redact_recall"] * r["redact_precision"] / max(1e-9, r["redact_recall"] + r["redact_precision"])
        res[bname] = r
        if args.save_preds:
            pdir = os.path.join(os.path.dirname(os.path.abspath(args.out)), "preds"); os.makedirs(pdir, exist_ok=True)
            json.dump(saved, open(f"{pdir}/{name.strip('/').replace('/', '__')[-60:]}__{bname}.json", "w"))
        print(f"{name[-50:]:50s} {bname:24s} " + " ".join(f"{k}={v:.3f}" for k, v in r.items()), flush=True)
    res["_meta"] = dict(seconds=time.time() - t0)
    results[name] = res
    json.dump(results, open(args.out, "w"), indent=1)
    del nlp; torch.cuda.empty_cache()
