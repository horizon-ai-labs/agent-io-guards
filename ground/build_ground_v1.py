"""Groundedness data v1 = v0 (response level) + CLAIM-LEVEL pairs derived from the same generations.

Each unsupported variant is a minimal edit of the supported response, so sentence diffs give claim labels:
  every sentence of the supported response -> supported (1); every sentence of a variant that does not occur in
  the supported response -> unsupported (0). Unchanged sentences of variants are skipped (they are duplicates).
v0 models trained on whole responses only and lost to MiniCheck on LLM-AggreFact (sentence-level claims).
env: V0=<ground_v0 dir> GEN=<glob ground_*.jsonl> OUT=<dir>
"""
import glob, json, os, random, re
import pandas as pd

V0, OUT = os.environ["V0"], os.environ["OUT"]; os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(1)
SPLIT = re.compile(r"(?<=[.!?。！？])\s+|\n+")


def sents(t):
    return [s.strip() for s in SPLIT.split(t) if len(s.strip()) > 15]


v0_eval = pd.read_parquet(f"{V0}/eval/synthetic_ground.parquet")
held = set(v0_eval.text)  # contexts held out in v0 stay held out
rows, stats = [], dict(sup=0, unsup=0, items=0)
for f in sorted(glob.glob(os.environ["GEN"])):
    for line in open(f):
        r = json.loads(line)
        ctx = (r["task"].strip() + "\n\n" + r["passage"]).strip() if r.get("task") else r["passage"]
        if ctx in held:
            continue
        stats["items"] += 1
        sup = [x for x in r["responses"] if x["label"] == 1]
        if not sup:
            continue
        ss = sents(sup[0]["text"]); sset = set(ss)
        for s in ss:
            rows.append(dict(text=r["passage"], text_pair=s, label=1, source="synthetic_claim/supported", lang=r["lang"])); stats["sup"] += 1
        for x in r["responses"]:
            if x["label"] == 0:
                for s in sents(x["text"]):
                    if s not in sset:
                        rows.append(dict(text=r["passage"], text_pair=s, label=0, source="synthetic_claim/unsupported", lang=r["lang"])); stats["unsup"] += 1
claims = pd.DataFrame(rows).drop_duplicates(["text", "text_pair"])
# balance: the supported side has more sentences; keep all unsupported, sample supported to 1:1
pos, neg = claims[claims.label == 1], claims[claims.label == 0]
claims = pd.concat([pos.sample(min(len(pos), len(neg)), random_state=0), neg])
tr0 = pd.read_parquet(f"{V0}/train.parquet"); va0 = pd.read_parquet(f"{V0}/val.parquet")
claims = claims.sample(frac=1.0, random_state=3)
nval = 1500
cols = ["text", "text_pair", "label", "source", "lang"]
pd.concat([tr0[cols], claims.iloc[nval:][cols]]).sample(frac=1.0, random_state=4).to_parquet(f"{OUT}/train.parquet")
pd.concat([va0[cols], claims.iloc[:nval][cols]]).to_parquet(f"{OUT}/val.parquet")
v0_eval.to_parquet(f"{OUT}/eval/synthetic_ground.parquet")
print(stats, "claims kept", len(claims), "train total", len(tr0) + len(claims) - nval, flush=True)
