"""Injection data v2.3 = v2.2 (soft = max(large, qwen)) + data-format task-hijack pairs (inj/gen_hijack3.py; soft = label).
python inj/build_v23.py V22_DIR OUT_DIR HIJACK3.jsonl [...]
Removes new texts that occur in any eval set (normalised exact match); 3% of new pairs go to val (both members together)."""
import glob, json, os, re, sys, zlib
import pandas as pd
v22, out, ins = sys.argv[1], sys.argv[2], sys.argv[3:]
os.makedirs(out, exist_ok=True)
norm = lambda s: re.sub(r"\W+", " ", str(s).lower()).strip()
EV = ["/fast/awuhrmann/agent-outputs/run-20260923-130458/65d4f827-c645-4a38-bf33-8d5ea76adb08/v2/eval", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".gated", "eval")]
ev = set()
for d in EV:
    for p in glob.glob(f"{d}/*.parquet") + glob.glob(f"/outputs/65d4f827-c645-4a38-bf33-8d5ea76adb08/v2/eval/*.parquet"):
        ev |= {norm(t) for t in pd.read_parquet(p, columns=["text"]).text}
LC = {"English": "en", "German": "de", "French": "fr", "Spanish": "es", "Portuguese": "pt", "Italian": "it", "Dutch": "nl", "Polish": "pl",
      "Russian": "ru", "Ukrainian": "uk", "Turkish": "tr", "Arabic": "ar", "Hindi": "hi", "Simplified Chinese": "zh", "Japanese": "ja",
      "Korean": "ko", "Vietnamese": "vi", "Indonesian": "id", "Thai": "th", "Czech": "cs", "Swedish": "sv", "Persian": "fa", "Hebrew": "he",
      "Bengali": "bn", "Romanian": "ro", "Greek": "el", "Hungarian": "hu", "Danish": "da", "Finnish": "fi", "Swahili": "sw", "Tagalog": "tl"}
rows = {"train": [], "val": []}; st = dict(pairs=0, dropped_eval_overlap=0)
for f in ins:
    for line in open(f):
        p = json.loads(line); st["pairs"] += 1
        if norm(p["legit"]) in ev or norm(p["hijacked"]) in ev:
            st["dropped_eval_overlap"] += 1; continue
        sp = "val" if zlib.crc32(p["hijacked"].encode()) % 100 < 3 else "train"
        lang = LC.get(p["lang"], "xx")
        rows[sp] += [dict(text=p["legit"], label=0, source="gen3_hijack/legit", kind="indirect", lang=lang, soft=0.0),
                     dict(text=p["hijacked"], label=1, source="gen3_hijack/hijacked", kind="indirect", lang=lang, soft=1.0)]
for sp in ["train", "val"]:
    base = pd.read_parquet(f"{v22}/{sp}.parquet")
    new = pd.DataFrame(rows[sp])
    df = pd.concat([base, new], ignore_index=True).sample(frac=1.0, random_state=0).reset_index(drop=True)
    df.to_parquet(f"{out}/{sp}.parquet"); st[f"n_{sp}"] = len(df); st[f"new_{sp}"] = len(new)
json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(st)
