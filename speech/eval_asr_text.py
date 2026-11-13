"""Punctuation + casing on REAL speech-recognizer text vs clean text (FLEURS test; speech/fleurs_asr.py output).
Systems: whisper (Whisper's own marks and casing), and our models on lowercased, mark-stripped input - once on the ASR words (with word
errors) and once on the gold words (clean reference). ASR words are aligned to gold words (difflib, case/punctuation-insensitive);
scoring is END-TO-END: a gold mark/capital on a word the recognizer got wrong is a miss, a prediction on an unaligned word a false alarm.
Punctuation F1 over . , ? : - (micro per language); casing F1 over Cap/UPPER (exact class). Mean over languages.
python speech/eval_asr_text.py OUT.json FLEURS_ASR.parquet joint:<truecase dir> ... punct:<punctuation dir> ..."""
import difflib, json, os, re, sys, unicodedata
import numpy as np, pandas as pd, torch
H = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(H, "..", "punct")); sys.path.insert(0, os.path.join(H, "..", "truecase"))
from common import units, NOSPACE, lower_keep_len
from common_tc import unit_cases, CASES
from transformers import AutoModelForTokenClassification, AutoTokenizer

out_f, inp = sys.argv[1], sys.argv[2]; specs = sys.argv[3:]
d = pd.read_parquet(inp)
norm = lambda w: re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", w).lower())


def prep(raw, lang):
    """punctuated cased text -> (clean cased text, unit ends, marks, cases, unit strings)"""
    c, e, l = units(raw, lang); nos = lang in NOSPACE
    cs = unit_cases(c, e, nos)
    us, prev = [], 0
    for x in e:
        s = x - 1 if nos else c.rfind(" ", 0, x) + 1
        us.append(c[max(s, prev):x]); prev = x
    return c, e, l, cs, us


G = [prep(r, l) for r, l in zip(d.gold_raw, d.lang)]
A = [prep(r, l) for r, l in zip(d.asr_raw, d.lang)]
ALIGN = []
for g, a in zip(G, A):
    m = {}
    for i, j, n in difflib.SequenceMatcher(None, [norm(u) for u in g[4]], [norm(u) for u in a[4]], autojunk=False).get_matching_blocks():
        for k in range(n):
            m[j + k] = i + k
    ALIGN.append(m)   # asr unit -> gold unit
print("aligned share of gold words:", round(np.mean([len(m) / max(1, len(g[4])) for m, g in zip(ALIGN, G)]), 4), flush=True)


def predict(spec, texts_ends):
    kind, path = spec.split(":", 1)
    tok = AutoTokenizer.from_pretrained(path); model = AutoModelForTokenClassification.from_pretrained(path).cuda().eval()
    id2 = model.config.id2label; res = []
    for text, ends in texts_ends:
        enc = tok(lower_keep_len(text), truncation=True, max_length=512, return_offsets_mapping=True, return_tensors="pt")
        offs = enc.pop("offset_mapping")[0].tolist()
        with torch.no_grad():
            pr = model(**{k: v.cuda() for k, v in enc.items()}).logits[0].argmax(-1).tolist()
        at = {b: id2[p] for (a, b), p in zip(offs, pr) if b > a}
        labs = [at.get(x, id2[0]) for x in ends]
        if kind == "joint":
            res.append(([x.split("|")[0] for x in labs], [CASES.index(x.split("|")[1]) for x in labs]))
        else:   # punctuation-only model: rule casing (first word, after . or ?)
            marks, cases, cap = [x if x in ".,?:-" else "0" for x in labs], [], True
            for mk in marks:
                cases.append(1 if cap else 0); cap = mk in (".", "?")
            res.append((marks, cases))
    del model; torch.cuda.empty_cache(); return res


def score(preds, use_asr):
    per = {}
    for idx, ((pm, pc), g) in enumerate(zip(preds, G)):
        lang = d.lang.iloc[idx]; s = per.setdefault(lang, np.zeros(6))   # punct tp fp fn, case tp fp fn
        if use_asr:
            m = ALIGN[idx]; hyp = {gi: (pm[j], pc[j]) for j, gi in m.items()}
            extra = [(pm[j], pc[j]) for j in range(len(pm)) if j not in m]   # predictions on misrecognized words
        else:
            hyp = {i: (pm[i], pc[i]) for i in range(len(pm))}; extra = []
        for i, (gm, gc) in enumerate(zip(g[2], g[3])):
            hm, hc = hyp.get(i, ("0", 0))
            if gm != "0":
                s[0 if hm == gm else 2] += 1
            if hm != "0" and hm != gm:
                s[1] += 1
            if gc != 0:
                s[3 if hc == gc else 5] += 1
            if hc != 0 and hc != gc:
                s[4] += 1
        for hm, hc in extra:
            s[1] += hm != "0"; s[4] += hc != 0
    f = lambda tp, fp, fn: 2 * tp / max(1, 2 * tp + fp + fn)
    pl = {l: dict(punct_f1=round(f(*v[:3]), 4), case_f1=round(f(*v[3:]), 4)) for l, v in per.items()}
    return dict(punct_f1=round(float(np.mean([x["punct_f1"] for x in pl.values()])), 4), case_f1=round(float(np.mean([x["case_f1"] for x in pl.values()])), 4), per_lang=pl)


res = {"whisper_native": score([(a[2], a[3]) for a in A], True)}
print("whisper_native", {k: v for k, v in res["whisper_native"].items() if k != "per_lang"}, flush=True)
for spec in specs:
    name = spec.split(":", 1)[0] + "_" + spec.rstrip("/").split("/")[-2][:8]
    res[name + "@asr"] = score(predict(spec, [(a[0], a[1]) for a in A]), True)
    res[name + "@clean"] = score(predict(spec, [(g[0], g[1]) for g in G]), False)
    for k in (name + "@asr", name + "@clean"):
        print(k, {x: y for x, y in res[k].items() if x != "per_lang"}, flush=True)
json.dump(res, open(out_f, "w"), indent=1)
