"""Real user prompts for a multilingual prompt task & complexity classifier (CPU job). Only the text of the first user message is kept;
all other WildChat fields (hashed IP, location, headers, model) are dropped.
- allenai/WildChat-1M (ODC-BY): first user turn of conversations not flagged toxic, 10-4000 characters;
- OpenAssistant/oasst2 (Apache-2.0): root prompter messages;
- CohereLabs/aya_dataset (Apache-2.0): instruction inputs.
Deduplicated (normalised text). Per-language caps keep English/Chinese from dominating. TEST = held-out WildChat prompts, up to 200 per
language for languages with >= 1,000 prompts (never used for training).
python prompt/prep_prompts.py OUT_DIR -> OUT_DIR/{train,test}_prompts.parquet (text, lang, source)"""
import ast, os, re, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()


def first_user(conv):
    try:
        c = conv if isinstance(conv, (list, tuple)) or hasattr(conv, "__len__") and not isinstance(conv, str) else ast.literal_eval(conv)
        for m in c:
            if m["role"] == "user":
                return m["content"]
    except Exception:
        return None


rows = []
for f in [f for f in HfApi().list_repo_files("allenai/WildChat-1M", repo_type="dataset") if f.endswith(".parquet")]:
    d = pd.read_parquet(hf_hub_download("allenai/WildChat-1M", f, repo_type="dataset"), columns=["conversation", "language", "toxic"])
    d = d[~d.toxic.astype(bool)]
    for conv, lang in zip(d.conversation, d.language):
        t = first_user(conv)
        if isinstance(t, str) and 10 <= len(t.strip()) <= 4000:
            rows.append((t.strip(), lang, "wildchat"))
    print(f, len(rows), flush=True)
o = pd.concat([pd.read_parquet(hf_hub_download("OpenAssistant/oasst2", f, repo_type="dataset")) for f in HfApi().list_repo_files("OpenAssistant/oasst2", repo_type="dataset")
               if f.endswith(".parquet") and "ready" not in f and "trees" not in f])
o = o[(o.role == "prompter") & o.parent_id.isna()]
rows += [(t.strip(), l, "oasst2") for t, l in zip(o.text, o.lang) if isinstance(t, str) and 10 <= len(t.strip()) <= 4000]
a = pd.concat([pd.read_parquet(hf_hub_download("CohereLabs/aya_dataset", f, repo_type="dataset")) for f in HfApi().list_repo_files("CohereLabs/aya_dataset", repo_type="dataset")
               if f.endswith(".parquet") and "train" in f])
rows += [(t.strip(), l, "aya") for t, l in zip(a.inputs, a.language) if isinstance(t, str) and 10 <= len(t.strip()) <= 4000]
df = pd.DataFrame(rows, columns=["text", "lang", "source"]); n0 = len(df)
df["k"] = df.text.map(norm); df = df[df.k.str.len() >= 5].drop_duplicates("k").sample(frac=1, random_state=0)
print("prompts", n0, "-> dedup", len(df), df.source.value_counts().to_dict(), flush=True)
wc = df[df.source == "wildchat"]; big = wc.lang.value_counts(); big = big[big >= 1000].index
test = wc[wc.lang.isin(big)].groupby("lang").head(200)
rest = df.drop(test.index)
CAP = {"English": 40000, "Chinese": 15000}
train = rest.groupby("lang").head(8000)
train = pd.concat([train[~train.lang.isin(CAP)]] + [rest[rest.lang == l].head(c) for l, c in CAP.items()]).drop_duplicates("k")
for n, x in [("train", train), ("test", test)]:
    x[["text", "lang", "source"]].reset_index(drop=True).to_parquet(f"{OUT}/{n}_prompts.parquet")
    print(n, len(x), x.lang.nunique(), "langs", x.source.value_counts().to_dict(), x.lang.value_counts().head(12).to_dict(), flush=True)
