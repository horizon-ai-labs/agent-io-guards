"""Evaluate language identifiers on FLORES-200 devtest (full sentences and 20-40 character prefixes).
python lid/eval_lid.py --models A,B --evals DIR --out results.json
Model specs: <hub id or dir> (our transformers classifiers, FLORES labels); papluca:<hub id> (20 ISO-639-1 labels, scored on
its 20 languages only); fasttext:<hub id> (fastText .bin, labels __label__<iso3>_<Script>; GlotLID / NLLB LID).
Chinese: zho_Hans, zho_Hant, cmn_Hani, zho_Hani all count as zho_Hani; twi_Latn counts as aka_Latn (Twi is a variety of Akan).
Metrics: accuracy, macro-F1 over languages, per-language accuracy; for papluca also ours on the same 20-language subset.
"""
import argparse, glob, json, os
import numpy as np, pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--models", required=True); ap.add_argument("--evals", required=True); ap.add_argument("--out", required=True)
ap.add_argument("--bs", type=int, default=256)
args = ap.parse_args()
EV = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in sorted(glob.glob(f"{args.evals}/*.parquet"))}
ZH = {"zho_Hans", "zho_Hant", "cmn_Hani", "zho_Hani", "cmn_Hans", "cmn_Hant"}
# merged labels (lid/merge_labels.py): Twi -> Akan; Arabic dialects -> arb_Arab; Dyula -> Bambara (applied to every model)
MERGED = {"twi_Latn": "aka_Latn", "dyu_Latn": "bam_Latn", **{k: "arb_Arab" for k in ["ars_Arab", "acm_Arab", "apc_Arab", "ary_Arab", "aeb_Arab", "arz_Arab", "acq_Arab", "ajp_Arab"]}}
canon = lambda l: "zho_Hani" if l in ZH else MERGED.get(l, l)
PAP = dict(ar="arb_Arab", bg="bul_Cyrl", de="deu_Latn", el="ell_Grek", en="eng_Latn", es="spa_Latn", fr="fra_Latn", hi="hin_Deva",
           it="ita_Latn", ja="jpn_Jpan", nl="nld_Latn", pl="pol_Latn", pt="por_Latn", ru="rus_Cyrl", sw="swh_Latn", th="tha_Thai",
           tr="tur_Latn", ur="urd_Arab", vi="vie_Latn", zh="zho_Hani")


def metrics(y, p):
    from sklearn.metrics import f1_score
    y = np.array([canon(x) for x in y]); p = np.array([canon(x) for x in p])
    per = {l: float((p[y == l] == l).mean()) for l in sorted(set(y))}
    return dict(n=len(y), acc=float((y == p).mean()), macro_f1=float(f1_score(y, p, labels=sorted(set(y)), average="macro")), per_lang=per)


def hf_predict(spec, texts, allowed=None):
    """allowed: optional set of labels; logits of all other labels are masked (user-supplied candidate languages)."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(spec); m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    id2 = m.config.id2label; out = []
    mask = None
    if allowed is not None:
        mask = torch.full((len(id2),), float("-inf")); mask[[i for i, l in id2.items() if canon(l) in allowed]] = 0; mask = mask.cuda()
    order = np.argsort([len(t) for t in texts]); pred = [None] * len(texts)
    with torch.no_grad():
        for s in range(0, len(texts), args.bs):
            b = order[s:s + args.bs]
            enc = tok([texts[i] for i in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
            lo = m(**{k: v.cuda() for k, v in enc.items() if k in ("input_ids", "attention_mask")}).logits.float()
            if mask is not None:
                lo = lo + mask
            for i, j in zip(b, lo.argmax(-1).tolist()):
                pred[i] = id2[j]
    return pred


def ft_predict(spec, texts):
    import fasttext
    from huggingface_hub import hf_hub_download
    m = fasttext.load_model(hf_hub_download(spec, "model.bin"))
    labs, _ = m.predict([t.replace("\n", " ") for t in texts], k=1)
    return [l[0].replace("__label__", "") for l in labs]


res = {}
for spec in args.models.split(","):
    kind, mid = (spec.split(":", 1) if ":" in spec else ("hf", spec))
    r = {}
    for name, df in EV.items():
        texts = df.text.tolist(); y = df.label.tolist()
        if kind == "papluca":
            sub = df[df.label.isin(set(PAP.values()))]
            p = [PAP.get(x, x) for x in hf_predict(mid, sub.text.tolist())]
            r[name] = metrics(sub.label.tolist(), p)
        elif kind == "fasttext":
            r[name] = metrics(y, ft_predict(mid, texts))
        else:
            p = hf_predict(mid, texts); r[name] = metrics(y, p)
            m = df.label.isin(set(PAP.values())).values
            r[name + "_papluca20"] = metrics(list(np.array(y)[m]), list(np.array(p)[m]))
            sub = df[m]   # same 20 languages, prediction restricted to papluca's 20 candidates (like-for-like with papluca)
            r[name + "_papluca20_restricted"] = metrics(sub.label.tolist(), hf_predict(mid, sub.text.tolist(), allowed=set(PAP.values())))
        print(spec, name, round(r[name]["acc"], 4), round(r[name]["macro_f1"], 4), flush=True)
    res[spec] = r
    json.dump(res, open(args.out, "w"), indent=1)
