"""results.json for the multilingual zero-shot classification leaderboard Space.  python release/zs_leaderboard/build.py"""
import json

bl = json.load(open("release/evals/zs_baselines.json")); nat = json.load(open("release/evals/zs_native_labels.json"))
SETS = [("MASSIVE", "massive", "intents, 16 langs"), ("MTOP", "mtop", "intents, 6 langs"), ("Banking77", "banking77", "intents, en"),
        ("CLINC150 §", "clinc", "intents, en"), ("SIB-200 §", "sib200", "topics, 16 langs"), ("AG News §", "agnews", "topics, en"),
        ("Yahoo §", "yahoo", "topics, en"), ("Emotion §", "emotion", "en"), ("SST-2 §", "sst2", "sentiment, en")]
MODELS = [("Horizon-Labs/multilingual-zeroshot-small", "141M", "yes", "zs_small"), ("Horizon-Labs/multilingual-zeroshot-base", "308M", "yes", "zs_base"),
          ("Horizon-Labs/multilingual-zeroshot-large", "568M", "yes", "zs_large"),
          ("MoritzLaurer/bge-m3-zeroshot-v2.0", "568M", "no (partly NC)", None), ("MoritzLaurer/bge-m3-zeroshot-v2.0-c", "568M", "yes", None),
          ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "278M", "no (XNLI/ANLI NC)", None),
          ("joeddav/xlm-roberta-large-xnli", "560M", "no (XNLI NC)", None), ("facebook/bart-large-mnli", "407M", "yes", None),
          ("MoritzLaurer/deberta-v3-base-zeroshot-v2.0", "184M", "no (partly NC)", None),
          ("MoritzLaurer/deberta-v3-large-zeroshot-v2.0", "435M", "no (partly NC)", None),
          ("MoritzLaurer/ModernBERT-base-zeroshot-v2.0", "149M", "no (partly NC)", None)]
rows = []
for mid, size, com, f in MODELS:
    e = json.load(open(f"release/evals/{f}.json")) if f else bl[mid]
    r = dict(model=mid, size=size, commercial_data=com)
    for name, key, _ in SETS:
        r[name] = e[key]["all"]["acc"] if key in e else None
    r["XNLI †"] = e["xnli"]["all"]["bacc"] if "xnli" in e else None
    n = nat.get(f.replace("zs_", "") if f else mid)
    r["MASSIVE native"] = n["massive"] if n else None; r["SIB-200 native"] = n["sib200"] if n else None
    intents = [r[x] for x in ["MASSIVE", "MTOP", "Banking77", "CLINC150 §"]]
    r["intent macro"] = sum(intents) / 4 if None not in intents else None
    rows.append(r)
sets = [(n, g) for n, k, g in SETS] + [("MASSIVE native", "labels in the text's language, 15 langs"), ("SIB-200 native", "15 langs"),
                                        ("XNLI †", "balanced acc., 12 langs")]
json.dump(dict(sets=sets, rows=rows, snapshot="2026-09-27"), open("release/zs_leaderboard/results.json", "w"), indent=1)
for r in sorted(rows, key=lambda r: -(r["intent macro"] or 0)):
    print(f"{r['model'][:55]:55s} {r['intent macro'] if r['intent macro'] is None else round(r['intent macro'],3)}")
