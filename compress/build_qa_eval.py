"""Multilingual RAG-style QA eval for prompt compression (eval only; never used for training).
XQuAD (12 langs, CC-BY-SA-4.0) and TyDi QA GoldP validation (9 langs; GoldP has no ja/th; Apache-2.0). Per language 200 questions; each context =
the gold paragraph + 4 paragraphs from OTHER articles (XQuAD: other paragraphs) of the same language, shuffled (gold position random), joined by blank lines.
python compress/build_qa_eval.py OUT.parquet"""
import random, sys
import pandas as pd
from huggingface_hub import hf_hub_download

out = sys.argv[1]; N, K = 200, 4
TYDI = {"arabic": "ar", "bengali": "bn", "english": "en", "finnish": "fi", "indonesian": "id", "japanese": "ja", "korean": "ko",
        "russian": "ru", "swahili": "sw", "telugu": "te", "thai": "th"}
rows = []


def build(df, set_name, lang, seed):
    rng = random.Random(seed)
    if "title" not in df:   # XQuAD has no titles: distractors are other paragraphs (rarely the same article)
        df = df.assign(title=df.context)
    paras = df.drop_duplicates("context")[["title", "context"]].values.tolist()
    q = df.sample(min(N, len(df)), random_state=seed)
    for _, r in q.iterrows():
        others = [c for t, c in paras if t != r.title]
        ctx = rng.sample(others, K) + [r.context]; rng.shuffle(ctx)
        rows.append(dict(set=set_name, lang=lang, id=r.id, question=r.question, answers=list(r.answers["text"]), context="\n\n".join(ctx),
                         gold_pos=ctx.index(r.context)))


for l in ["ar", "de", "el", "en", "es", "hi", "ro", "ru", "th", "tr", "vi", "zh"]:
    df = pd.read_parquet(hf_hub_download("google/xquad", f"xquad.{l}/validation-00000-of-00001.parquet", repo_type="dataset"))
    build(df, "xquad", l, 1000 + len(rows))
t = pd.read_parquet(hf_hub_download("google-research-datasets/tydiqa", "secondary_task/validation-00000-of-00001.parquet", repo_type="dataset"))
t["language"] = t.id.str.split("-").str[0]
for name, l in TYDI.items():
    build(t[t.language == name], "tydiqa", l, 2000 + len(rows))
d = pd.DataFrame(rows); d.to_parquet(out, index=False)
print(d.groupby("set").lang.nunique().to_dict(), len(d), "context words median", int(d.context.str.split().str.len().median()))
