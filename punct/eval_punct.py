"""Evaluate punctuation-restoration token classifiers on punct/build_evals.py sets.
python punct/eval_punct.py OUT.json EVAL_DIR MODEL [MODEL ...]     (MODEL = local dir or Hub id; labels "0 . , ? - :")
Every model gets the same input: the paragraph with target marks removed (punct/common.py units), cased and lowercased,
in chunks of 150 words (250 CJK characters). A word's prediction is the label of its last token (as in
deepmultilingualpunctuation). Metrics per set and language: punctuation F1 (micro over the 5 marks: a predicted mark is
correct only if it is the gold mark) and per-mark F1.
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch
from transformers import AutoModelForTokenClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import LABELS, NOSPACE, units, lower_keep_len, unit_preds

out_f, ev = sys.argv[1:3]; models = sys.argv[3:]
KREDOR = {"eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "bul_Cyrl", "ita_Latn", "pol_Latn", "nld_Latn", "ces_Latn", "por_Latn",
          "slk_Latn", "slv_Latn"}
sets = {os.path.basename(p)[:-8]: pd.read_parquet(f"{ev}/{os.path.basename(p)}") for p in sorted(os.listdir(ev)) if p.endswith(".parquet")}
items = []   # (set, lang, clean chunk, ends, gold labels)
for name, df in sets.items():
    for text, lang in zip(df.text, df.lang):
        nos = lang.startswith(("zho", "jpn", "yue")) or lang in NOSPACE
        c, e, l = units(text, "cmn_Hani" if lang.startswith(("zho", "yue")) else lang)
        cap = 250 if nos else 150
        for a in range(0, len(e), cap):
            off = e[a - 1] + (0 if nos else 1) if a else 0
            ee = e[a:a + cap]
            items.append((name, lang, c[off:ee[-1]], [x - off for x in ee], l[a:a + cap]))
print("chunks", len(items), flush=True)
res = json.load(open(out_f)) if os.path.exists(out_f) else {}
for mname in models:
    tok = AutoTokenizer.from_pretrained(mname)
    model = AutoModelForTokenClassification.from_pretrained(mname, torch_dtype=torch.bfloat16).cuda().eval()
    id2 = {int(k): v for k, v in model.config.id2label.items()}
    remap = np.array([LABELS.index(id2[i]) if id2[i] in LABELS else 0 for i in range(len(id2))])
    r = {}; t0 = time.time()
    for mode in ["cased", "lower"]:
        texts = [lower_keep_len(it[2]) if mode == "lower" else it[2] for it in items]
        order = np.argsort([len(t) for t in texts]); preds = [None] * len(items)
        for s in range(0, len(order), 64):
            idx = order[s:s + 64]
            enc = tok([texts[i] for i in idx], truncation=True, max_length=512, padding=True, return_offsets_mapping=True, return_tensors="pt")
            offs = enc.pop("offset_mapping").tolist()
            with torch.no_grad():
                pr = model(**{k: v.cuda() for k, v in enc.items()}).logits.argmax(-1).cpu().numpy()
            for j, i in enumerate(idx):
                preds[i] = unit_preds(offs[j], remap[pr[j]].tolist(), items[i][3])
        cnt = {}
        for it, p in zip(items, preds):
            k = (it[0], it[1]); c = cnt.setdefault(k, np.zeros((6, 6), np.int64))
            for g, q in zip(it[4], p):
                c[LABELS.index(g), LABELS.index(q)] += 1
        for (sname, lang), c in cnt.items():
            tp = sum(c[i, i] for i in range(1, 6)); fp = c[:, 1:].sum() - tp; fn = c[1:, :].sum() - tp
            d = r.setdefault(sname, {}).setdefault(lang, {})
            d[mode] = dict(f1=2 * tp / max(1, 2 * tp + fp + fn), conf=c.tolist(),
                           **{LABELS[i]: 2 * c[i, i] / max(1, c[i, :].sum() + c[:, i].sum()) for i in range(1, 6)})
    summ = {}
    for sname, langs in r.items():
        for mode in ["cased", "lower"]:
            for sub, ls in [("all", list(langs)), ("kredor12", [l for l in langs if l in KREDOR])]:
                if ls:
                    summ[f"{sname}/{mode}/{sub}"] = float(np.mean([langs[l][mode]["f1"] for l in ls]))
            pooled = sum(np.array(langs[l][mode]["conf"]) for l in langs)
            for i in (1, 2, 3):
                summ[f"{sname}/{mode}/pooled_{LABELS[i]}"] = float(2 * pooled[i, i] / max(1, pooled[i, :].sum() + pooled[:, i].sum()))
    r["_summary"] = summ; res[mname] = r
    print(mname, f"{time.time() - t0:.0f}s", json.dumps({k: round(v, 4) for k, v in summ.items()}), flush=True)
    json.dump(res, open(out_f, "w"))
    del model; torch.cuda.empty_cache()
