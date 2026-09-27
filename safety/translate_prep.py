"""Items for machine translation (content-safety v4).  python safety/translate_prep.py POOL1 POOL2 OUT.json [PER_LANG]
Why: our toxicity data (Civil Comments) and most red-team prompts are English; textdetox recall (14 languages) is .47
and Aya red-teaming recall .77. Items: Civil Comments train shard 1 (not used in v3; CC0) toxic (>= 0.5, soft label =
annotator fraction, subtype categories) and clean (0.0), plus teacher-labelled pool prompts: harmful (teacher >= 0.5),
benign-but-edgy (OR-Bench, benign framings, WildChat with P(controversial) >= 0.2) and random benign. Each item gets one
target language; every language gets a different set of source items.
"""
import json, random, sys
import pandas as pd
from huggingface_hub import hf_hub_download

P1, P2, OUT = sys.argv[1:4]
PER = int(sys.argv[4]) if len(sys.argv) > 4 else 4800
LANGS = ["Portuguese", "Russian", "Ukrainian", "Polish", "Czech", "Swedish", "Turkish", "Hebrew", "Serbian", "Tagalog", "Amharic",
         "Tatar", "German", "Spanish", "French", "Arabic", "Hindi", "Chinese (Simplified)", "Japanese", "Italian"]
rng = random.Random(5)
n = PER * len(LANGS)
cc = pd.read_parquet(hf_hub_download("google/civil_comments", "data/train-00001-of-00002.parquet", repo_type="dataset"))
cc = cc[cc.text.str.len().between(10, 1500)]
SUB = {"obscene": "profanity", "threat": "violence", "insult": "harassment", "identity_attack": "hate", "sexual_explicit": "sexual"}
civ = []
for part in [cc[cc.toxicity >= 0.5].sample(n=n // 4, random_state=1), cc[cc.toxicity == 0].sample(n=n // 4, random_state=1)]:
    civ += [dict(text_en=r.text, source="civil", soft=float(r.toxicity), cats=[c for k, c in SUB.items() if getattr(r, k) >= 0.5]) for r in part.itertuples()]
pool = pd.concat([pd.read_parquet(P1), pd.read_parquet(P2)])
pool = pool[(pool.kind == "prompt") & pool.text.str.len().between(10, 1500)]
t = pool.p_unsafe + 0.5 * pool.p_contro
harm = pool[t >= 0.5]; edgy = pool[(t < 0.5) & (pool.source.str.contains("orbench|gen2_framing/benign") | (pool.p_contro >= 0.2))]
benign = pool[(t < 0.2) & ~pool.index.isin(edgy.index)]
k = n // 2
parts = [harm.sample(n=min(len(harm), int(k * 0.5)), random_state=1), edgy.sample(n=min(len(edgy), int(k * 0.3)), random_state=1),
         benign.sample(n=int(k * 0.2), random_state=1)]
pr = [dict(text_en=r.text, source="pool/" + r.source, soft=None, cats=[]) for d in parts for r in d.itertuples()]
items = civ + pr
rng.shuffle(items)
for i, it in enumerate(items):
    it["lang"] = LANGS[i % len(LANGS)]
json.dump(items, open(OUT, "w"), ensure_ascii=False)
print("items", len(items), "civil", len(civ), "pool", len(pr), "harm/edgy/benign", [len(d) for d in parts])
