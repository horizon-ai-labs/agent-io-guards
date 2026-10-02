"""Evaluate gibberish detectors.   python gib/eval_gib.py --models A,B --test TEST.parquet --flores FLORES.parquet --hand gib/handset.json --out R.json
- flores: FLORES-200 devtest sentences (professional translations, all clean), 150 per language: share predicted clean
  (= 1 - false-positive rate), per language and mean.
- test: our held-out synthetic test set (4 classes): accuracy and macro-F1.
- hand: 30 hand-written examples (10 languages-ish, 4 classes) written by us: accuracy.
Labels mapped by name (clean / mild gibberish / word salad / noise).
"""
import argparse, json, os
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
ap = argparse.ArgumentParser()
for a in ["--models", "--test", "--flores", "--hand", "--out"]:
    ap.add_argument(a, required=True)
args = ap.parse_args()
C4 = ["clean", "mild gibberish", "word salad", "noise"]
test = pd.read_parquet(args.test); hand = pd.DataFrame(json.load(open(args.hand)), columns=["text", "label"])
fl = pd.read_parquet(args.flores); fl = fl.groupby("label").head(150)


def macro_f1(y, p):
    f = []
    for c in C4:
        tp = ((p == c) & (y == c)).sum(); f.append(0.0 if tp == 0 else 2 * tp / (2 * tp + ((p == c) & (y != c)).sum() + ((p != c) & (y == c)).sum()))
    return float(np.mean(f))


res = json.load(open(args.out)) if os.path.exists(args.out) else {}
for spec in args.models.split(","):
    tok = AutoTokenizer.from_pretrained(spec); m = AutoModelForSequenceClassification.from_pretrained(spec, torch_dtype=torch.bfloat16).cuda().eval()
    labs = [m.config.id2label[i].lower() for i in range(m.config.num_labels)]

    @torch.no_grad()
    def pred(texts):
        out = [None] * len(texts); order = np.argsort([len(t) for t in texts])
        for s in range(0, len(texts), 128):
            b = order[s:s + 128]
            e = tok([texts[i] for i in b], truncation=True, max_length=256, padding=True, return_tensors="pt")
            lo = m(**{k: v.cuda() for k, v in e.items() if k in ("input_ids", "attention_mask")}).logits
            for i, j in zip(b, lo.argmax(-1).tolist()):
                out[i] = labs[j]
        return np.array(out)
    r = {}
    p = pred(fl.text.tolist()); fl_ok = pd.Series(p == "clean", index=fl.index).groupby(fl.label).mean()
    r["flores_clean_rate_mean"] = float(fl_ok.mean()); r["flores_clean_rate_worst10"] = fl_ok.sort_values().head(10).round(3).to_dict()
    r["flores_clean_rate_eng"] = float(fl_ok.get("eng_Latn", np.nan))
    pt = pred(test.text.tolist()); r["test_acc"] = float((pt == test.label.values).mean()); r["test_macro_f1"] = macro_f1(test.label.values, pt)
    r["test_by_label"] = {c: float((pt[test.label.values == c] == c).mean()) for c in C4}
    ph = pred(hand.text.tolist()); r["hand_acc"] = float((ph == hand.label.values).mean()); r["hand_errors"] = [(t, g, q) for t, g, q in zip(hand.text, hand.label, ph) if g != q]
    print(spec, {k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items() if not isinstance(v, (dict, list))}, flush=True)
    res[spec] = r; json.dump(res, open(args.out, "w"), indent=1, ensure_ascii=False)
