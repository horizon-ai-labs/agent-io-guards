"""Evaluate emotion classifiers.   python emotion/eval_emo.py --models A,B --evals emotion/evals --out R.json
- goemo: go_emotions test, 28 labels (only models with the 28 GoEmotions labels): macro-F1 at 0.5 and with per-label
  thresholds tuned on go_emotions validation.
- brighter_<lang>: BRIGHTER test, 6 Ekman emotions, multi-label. Every model's labels are grouped into the 6 emotions
  (GoEmotions labels via the GoEmotions paper's Ekman mapping; other label sets by name, see MAP); an emotion's score is
  the max over its member labels. Thresholds per model, language and emotion, tuned on that language's BRIGHTER dev set
  (pooled dev of all languages if the language's dev has < 5 positives). Macro-F1 over the emotions that are annotated for
  the language (and have >= 10 test positives) and that the model can predict; `covered` lists the emotions a model
  supports. _brighter_mean_f1: mean over languages; _brighter_mean_f1_4: same over anger, fear, joy, sadness only (the
  emotions every compared model covers).
Scores: sigmoid for multi-label models, softmax for single-label ones.
"""
import argparse, glob, json, os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import GO, EKMAN, E6, best_thresholds, f1

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=64); ap.add_argument("--max_len", type=int, default=512)
args = ap.parse_args()
MAP = {**{g: e for e, gs in EKMAN.items() for g in gs}, "frustration": "anger", "contempt": "disgust"}   # label -> Ekman emotion

goe_t, goe_d = pd.read_parquet(f"{args.evals}/goemo_test.parquet"), pd.read_parquet(f"{args.evals}/goemo_dev.parquet")
BT = {os.path.basename(p)[:-13]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/brighter/*_test.parquet"))}
BDL = {os.path.basename(p)[:-12]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/brighter/*_dev.parquet"))}
BD = pd.concat([d.assign(lang=l) for l, d in BDL.items()], ignore_index=True)
GRID = np.concatenate([[0.002, 0.005, 0.01], np.arange(0.02, 0.96, 0.02)])


def scores(m, tok, texts, multi):
    order = np.argsort([len(t) for t in texts]); S = np.zeros((len(texts), m.config.num_labels))
    with torch.no_grad():
        for s in range(0, len(texts), args.bs):
            b = order[s:s + args.bs]
            enc = tok([texts[i] for i in b], truncation=True, max_length=args.max_len, padding=True, return_tensors="pt")
            lo = m(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.float()
            S[b] = (torch.sigmoid(lo) if multi else torch.softmax(lo, -1)).cpu().numpy()
    return S


def ekman(S, labels):
    E = np.zeros((len(S), 6)); cov = []
    for j, e in enumerate(E6):
        idx = [i for i, l in enumerate(labels) if MAP.get(l.lower()) == e]
        if idx:
            E[:, j] = S[:, idx].max(1); cov.append(e)
    return E, cov


res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec)
    m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    labels = [m.config.id2label[i] for i in range(m.config.num_labels)]
    multi = m.config.problem_type == "multi_label_classification" or os.environ.get("FORCE_MULTI") == "1"
    r = dict(labels=labels, multi=multi)
    if sorted(l.lower() for l in labels) == sorted(GO):
        col = [labels.index(g) if g in labels else [l.lower() for l in labels].index(g) for g in GO]
        St, Sd = scores(m, tok, goe_t.text.tolist(), multi)[:, col], scores(m, tok, goe_d.text.tolist(), multi)[:, col]
        Yt = np.array([[g in ls for g in GO] for ls in goe_t.labels]); Yd = np.array([[g in ls for g in GO] for ls in goe_d.labels])
        thr = best_thresholds(Yd, Sd)
        r["goemo"] = dict(macro_f1_05=float(np.mean([f1(Yt[:, j], St[:, j] >= 0.5) for j in range(28)])),
                          macro_f1_tuned=float(np.mean([f1(Yt[:, j], St[:, j] >= thr[j]) for j in range(28)])),
                          per_label_tuned={g: f1(Yt[:, j], St[:, j] >= thr[j]) for j, g in enumerate(GO)}, thresholds=dict(zip(GO, thr.tolist())),
                          dev_macro_f1_tuned_insample=float(np.mean([f1(Yd[:, j], Sd[:, j] >= thr[j]) for j in range(28)])))   # for model selection
        print(spec, "goemo", round(r["goemo"]["macro_f1_05"], 4), round(r["goemo"]["macro_f1_tuned"], 4), flush=True)
    Ed, cov = ekman(scores(m, tok, BD.text.tolist(), multi), labels)
    r["covered"] = cov

    def tune(mask, j, e):
        y = BD[e].values.astype(float); ok = mask & ~np.isnan(y) & (y >= 0)
        if ok.sum() == 0 or y[ok].sum() < 5:
            return None
        return float(best_thresholds(y[ok][:, None].astype(bool), Ed[ok][:, [j]], GRID)[0])
    pooled = {e: tune(np.ones(len(BD), bool), j, e) for j, e in enumerate(E6) if e in cov}
    dev_f = []   # BRIGHTER dev macro-F1 with in-sample thresholds (for model selection; never the test sets)
    for lang in BDL:
        mk = (BD.lang == lang).values; fs = []
        for j, e in enumerate(E6):
            y = BD[e].values.astype(float)[mk]
            if e in cov and not np.isnan(y).any() and (y >= 0).all() and y.sum() >= 5:
                t = tune(mk, j, e); fs.append(f1(y.astype(bool), Ed[mk][:, j] >= t))
        if fs:
            dev_f.append(np.mean(fs))
    r["_brighter_dev_insample"] = float(np.mean(dev_f))
    means, means4 = [], []
    for lang, df in BT.items():
        E, _ = ekman(scores(m, tok, df.text.tolist(), multi), labels)
        per, thr = {}, {}
        for j, e in enumerate(E6):
            y = df[e].values.astype(float)
            if e not in cov or np.isnan(y).any() or (y < 0).any() or y.sum() < 10:
                continue
            thr[e] = tune((BD.lang == lang).values, j, e) or pooled[e] or 0.5
            per[e] = f1(y.astype(bool), E[:, j] >= thr[e])
        p4 = [per[e] for e in ("anger", "fear", "joy", "sadness") if e in per]
        r[f"brighter_{lang}"] = dict(n=len(df), macro_f1=float(np.mean(list(per.values()))) if per else None,
                                     macro_f1_4=float(np.mean(p4)) if p4 else None, per_emotion=per, thresholds=thr)
        if per:
            means.append(r[f"brighter_{lang}"]["macro_f1"])
        if p4:
            means4.append(r[f"brighter_{lang}"]["macro_f1_4"])
    r["_brighter_mean_f1"] = float(np.mean(means)); r["_brighter_mean_f1_4"] = float(np.mean(means4))
    print(spec, "covered", len(cov), "BRIGHTER mean", round(r["_brighter_mean_f1"], 4), "4-emotion mean", round(r["_brighter_mean_f1_4"], 4), flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1)
