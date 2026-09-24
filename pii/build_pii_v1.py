"""PII data v1 = v0 + Qwen-generated documents (pii/gen_pii.py) + their no-PII look-alike negatives.
env: V0=<pii_v0 dir> GEN=<glob of pii_gen_*.jsonl> OUT=<dir>.  5% of generated docs -> eval/synthetic_pii.parquet."""
import glob, json, os, random, re
import pandas as pd
V0, OUT = os.environ["V0"], os.environ["OUT"]
os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(3)
LANG = {"English": "en", "German": "de", "French": "fr", "Spanish": "es", "Portuguese": "pt", "Italian": "it", "Dutch": "nl",
        "Polish": "pl", "Russian": "ru", "Ukrainian": "uk", "Turkish": "tr", "Arabic": "ar", "Hindi": "hi", "Simplified Chinese": "zh",
        "Japanese": "ja", "Korean": "ko", "Vietnamese": "vi", "Indonesian": "id", "Thai": "th", "Czech": "cs", "Swedish": "sv",
        "Persian": "fa", "Hebrew": "he", "Bengali": "bn", "Romanian": "ro", "Greek": "el", "Hungarian": "hu", "Danish": "da",
        "Finnish": "fi", "Swahili": "sw", "Tagalog": "tl", "Urdu": "ur", "Malay": "ms", "Tamil": "ta", "Serbian": "sr",
        "Bulgarian": "bg", "Norwegian": "no"}
BANKISH = re.compile(r"(swift|bic|iban|routing|blz|sort code)", re.I)
tr, ev, stats = [], [], dict(docs=0, neg=0, relabel=0)
for f in sorted(glob.glob(os.environ["GEN"])):
    for line in open(f):
        r = json.loads(line)
        if not r.get("ok"):
            continue
        lang = LANG.get(r["args"]["lang"], "xx")
        text, spans = r["text"], []
        for s, e, l in r["spans"]:
            # the generator sometimes tags SWIFT/BIC/routing codes as SECRET: look at the 30 chars before the value
            if l == "SECRET" and BANKISH.search(text[max(0, s - 30):s]):
                l = "BANK_ACCOUNT"; stats["relabel"] += 1
            spans.append([s, e, l])
        rows = [dict(text=text, spans_json=json.dumps(spans), source="synthetic_pii", lang=lang)]
        if len(r.get("negative", "")) > 40:
            rows.append(dict(text=r["negative"], spans_json="[]", source="synthetic_pii_negative", lang=lang)); stats["neg"] += 1
        stats["docs"] += 1
        (ev if rng.random() < 0.05 else tr).extend(rows)
print(stats, len(tr), len(ev), flush=True)
t0 = pd.read_parquet(f"{V0}/train.parquet"); v0 = pd.read_parquet(f"{V0}/val.parquet")
new = pd.DataFrame(tr).drop_duplicates("text")
cols = ["text", "spans_json", "source", "lang"]
nval = 800
pd.concat([t0[cols], new.iloc[nval:][cols]]).sample(frac=1.0, random_state=4).to_parquet(f"{OUT}/train.parquet")
pd.concat([v0[cols], new.iloc[:nval][cols]]).to_parquet(f"{OUT}/val.parquet")
for p in glob.glob(f"{V0}/eval/*.parquet"):
    pd.read_parquet(p).to_parquet(f"{OUT}/eval/{os.path.basename(p)}")
pd.DataFrame(ev).to_parquet(f"{OUT}/eval/synthetic_pii.parquet")
print("train", len(t0) + len(new) - nval, flush=True)
