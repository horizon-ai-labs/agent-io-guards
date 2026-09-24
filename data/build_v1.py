"""Data v1 = data v0 + LLM-generated synthetic data (gen/gen_v1.py) + more jailbreak sources.

env: V0=<dir with v0 train/val/eval>  GEN=<glob of gen jsonl files>  OUT=<dir>
Generated docs are split by generation call: 8% go to a held-out eval set
(synthetic_docs_test / synthetic_direct_test), so the three variants of a document never straddle splits.
"""
import glob, json, os, random, re, shutil
import pandas as pd
from datasets import load_dataset

V0, OUT = os.environ["V0"], os.environ["OUT"]
GEN = sorted(glob.glob(os.environ["GEN"]))
os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(7)
LANG = {"English": "en", "German": "de", "French": "fr", "Spanish": "es", "Portuguese": "pt", "Italian": "it", "Dutch": "nl",
        "Polish": "pl", "Russian": "ru", "Ukrainian": "uk", "Turkish": "tr", "Arabic": "ar", "Hindi": "hi",
        "Simplified Chinese": "zh", "Japanese": "ja", "Korean": "ko", "Vietnamese": "vi", "Indonesian": "id", "Thai": "th",
        "Czech": "cs", "Swedish": "sv", "Persian": "fa", "Hebrew": "he", "Bengali": "bn", "Romanian": "ro", "Greek": "el",
        "Hungarian": "hu", "Danish": "da", "Finnish": "fi", "Swahili": "sw", "Tagalog": "tl"}


def key(t):
    # full normalized text: a clean doc and its injected twin share long prefixes
    return re.sub(r"\W+", " ", t.lower()).strip()


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def rec(text, label, source, kind="direct", lang="en"):
    return dict(text=str(text).strip(), label=int(label), source=source, kind=kind, lang=lang)


def strs(x):
    return [s for s in (x or []) if isinstance(s, str) and len(s.strip()) > 8]


new_train, ev_docs, ev_direct = [], [], []
stats = dict(calls=0, parsed=0, bad_inj=0)
for f in GEN:
    for line in open(f):
        r = json.loads(line)
        stats["calls"] += 1
        p, a = r["parsed"], r["args"]
        if not isinstance(p, dict):
            continue
        stats["parsed"] += 1
        lang = LANG.get(a.get("lang"), "xx")
        held = rng.random() < 0.08
        out = []
        if r["task"] == "docs":
            clean, inj, span, hard = (p.get(k) for k in ("clean", "injected", "injection_span", "hard_benign"))
            if isinstance(clean, str) and len(clean) > 50:
                out.append(rec(clean, 0, "gen_doc/clean", "indirect", lang))
            if isinstance(hard, str) and len(hard) > 50:
                out.append(rec(hard, 0, "gen_doc/hard_benign", "indirect", lang))
            ok = isinstance(inj, str) and len(inj) > 50 and norm(inj) != norm(clean or "")
            # the injected doc must actually contain (most of) the declared injection
            if ok and isinstance(span, str) and len(span) > 10:
                ok = norm(span)[:60] in norm(inj)
            if ok:
                out.append(rec(inj, 1, "gen_doc/injected", "indirect", lang))
            else:
                stats["bad_inj"] += 1
            (ev_docs if held else new_train).extend(out)
        else:
            out += [rec(t, 1, "gen_direct/attack", "direct", lang) for t in strs(p.get("attacks"))]
            out += [rec(t, 0, "gen_direct/benign", "direct", lang) for t in strs(p.get("benign"))]
            (ev_direct if held else new_train).extend(out)
print(stats, "gen train rows", len(new_train), "held docs", len(ev_docs), "held direct", len(ev_direct), flush=True)

# more jailbreak data: TrustAIRLab older snapshot, JailbreakV templates (+ their plain harmful queries as 0),
# NVIDIA PAIR-style persuasion jailbreaks
for r in load_dataset("TrustAIRLab/in-the-wild-jailbreak-prompts", "jailbreak_2023_05_07", split="train"):
    if r["prompt"]:
        new_train.append(rec(r["prompt"], 1, "trustairlab/jailbreak"))
jv = list(load_dataset("JailbreakV-28K/JailBreakV-28k", "JailBreakV_28K", split="JailBreakV_28K"))
rng.shuffle(jv)
seen_rt = set()
# only text jailbreak formats: in SD/typo/figstep/Logic the attack lives in the image and the text is a plain request
jv = [r for r in jv if r["format"] in ("Template", "Persuade")]
for r in jv[:6000]:
    new_train.append(rec(r["jailbreak_query"], 1, f"jailbreakv/{r['format']}"))
for r in load_dataset("JailbreakV-28K/JailBreakV-28k", "RedTeam_2K", split="RedTeam_2K"):
    new_train.append(rec(r["question"] if "question" in r else r.get("redteam_query", ""), 0, "jailbreakv/redteam_plain"))
nv = list(load_dataset("nvidia/Nemotron-RL-Jailbreak-Robustness-v1", split="train"))
rng.shuffle(nv)
for r in nv[:3000]:
    msgs = (r["responses_create_params"] or {}).get("input") or []
    u = [m["content"] for m in msgs if m.get("role") == "user" and isinstance(m.get("content"), str)]
    if u:
        new_train.append(rec(u[-1], 1, "nvidia_jailbreak"))

# ---------------------------------------------------------------- merge with v0
tr0 = pd.read_parquet(f"{V0}/train.parquet"); va0 = pd.read_parquet(f"{V0}/val.parquet")
evals = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in glob.glob(f"{V0}/eval/*.parquet")}
evals["synthetic_docs_test"] = pd.DataFrame(ev_docs).drop_duplicates("text")
evals["synthetic_direct_test"] = pd.DataFrame(ev_direct).drop_duplicates("text")
eval_keys = set()
for df in evals.values():
    eval_keys |= set(df.text.map(key))
new = pd.DataFrame(new_train)
new = new[new.text.str.len() > 3]
new["k"] = new.text.map(key)
before = len(new)
old_keys = set(tr0.text.map(key)) | set(va0.text.map(key))
new = new[~new.k.isin(eval_keys) & ~new.k.isin(old_keys)]
conf = new.groupby("k").label.nunique()
new = new[~new.k.isin(set(conf[conf > 1].index))].drop_duplicates("k")
print("new rows kept", len(new), "of", before, flush=True)
new = new.sample(frac=1.0, random_state=1)
nval = int(0.03 * len(new))
cols = ["text", "label", "source", "kind", "lang"]
tr = pd.concat([tr0, new.iloc[nval:][cols]]).sample(frac=1.0, random_state=2).reset_index(drop=True)
va = pd.concat([va0, new.iloc[:nval][cols]]).reset_index(drop=True)
tr.to_parquet(f"{OUT}/train.parquet"); va.to_parquet(f"{OUT}/val.parquet")
for n, df in evals.items():
    df[cols].to_parquet(f"{OUT}/eval/{n}.parquet")
summary = dict(train=len(tr), val=len(va), train_pos=int(tr.label.sum()),
               by_source=tr.groupby("source").label.agg(["count", "sum"]).reset_index().values.tolist(),
               by_lang=tr.lang.value_counts().head(40).to_dict(),
               evals={k: [len(v), int(v.label.sum())] for k, v in evals.items()})
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, default=str)[:3000])
