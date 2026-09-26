"""External content-safety benchmarks (evaluation only, never trained on).  python safety/build_safety_evals.py OUTDIR

OUTDIR/<name>.parquet with columns text, text_pair ("" = prompt only), label (1 = unsafe / toxic), lang.
  polyguard_prompt / polyguard_response  PolyGuardPrompts test (CC-BY-4.0; WildGuardTest translated into 17 languages),
                                         400 prompts / 300 prompt-response pairs per language (human harm labels)
  beavertails_response  BeaverTails 30k test (CC-BY-NC-4.0, eval only): 3021 prompt-response pairs, label = not is_safe
  beavertails_unseen    the 1894 of those whose prompt does not occur in our v0 training data (BeaverTails and Aegis
                        both take prompts from Anthropic HH red-team; 1127 overlap); needs env V0=<safety_v0 dir>
                        (normalised exact match against V0/train.parquet)
  toxicchat             ToxicChat 0124 test (CC-BY-NC-4.0, eval only): 5083 real user prompts, label = toxicity
  openai_mod            OpenAI moderation evaluation set (MIT): 1680 texts, label = any category flagged
  xstest                XSTest (CC-BY-4.0): 250 safe prompts that look unsafe + 200 unsafe contrasts (over-blocking)
  simplesafety          SimpleSafetyTests (CC-BY-2.0): 100 clearly unsafe prompts (recall)
  aya_redteaming        Aya red-teaming (Apache-2.0): harmful prompts written by native speakers, 8 languages, up to 300
                        each, all unsafe (recall)
  textdetox             textdetox multilingual toxicity (OpenRAIL++, eval only): 400 per language, 50% toxic
"""
import gzip, json, os, random, re, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)
dl = lambda r, f: hf_hub_download(r, f, repo_type="dataset")


def save(name, rows):
    df = pd.DataFrame(rows, columns=["text", "text_pair", "label", "lang"])
    df.to_parquet(f"{OUT}/{name}.parquet")
    print(name, len(df), "unsafe rate", round(df.label.mean(), 3), "langs", df.lang.nunique(), flush=True)


def sample(df, n):
    return df.sample(n=min(n, len(df)), random_state=0)


PGL = dict(English="en", Hindi="hi", French="fr", Italian="it", German="de", Portuguese="pt", Thai="th", Spanish="es", Czech="cs",
           Swedish="sv", Chinese="zh", Arabic="ar", Dutch="nl", Korean="ko", Polish="pl", Russian="ru", Japanese="ja")
p = pd.read_parquet(dl("ToxicityPrompts/PolyGuardPrompts", "data/test-00000-of-00001.parquet"))
rows = []
for lang, g in p[p.prompt_harm_label.isin(["harmful", "unharmful"])].groupby("language"):
    rows += [(r.prompt, "", int(r.prompt_harm_label == "harmful"), PGL[lang]) for r in sample(g, 400).itertuples()]
save("polyguard_prompt", rows)
rows = []
q = p[p.response_harm_label.isin(["harmful", "unharmful"]) & p.response.fillna("").str.strip().astype(bool)]
for lang, g in q.groupby("language"):
    rows += [(r.prompt, r.response, int(r.response_harm_label == "harmful"), PGL[lang]) for r in sample(g, 300).itertuples()]
save("polyguard_response", rows)

b = [json.loads(x) for x in gzip.open(dl("PKU-Alignment/BeaverTails", "round0/30k/test.jsonl.gz"), "rt")]
save("beavertails_response", [(r["prompt"], r["response"], int(not r["is_safe"]), "en") for r in b])
if os.environ.get("V0"):
    norm = lambda z: re.sub(r"\s+", " ", str(z).strip().lower())
    seen = set(pd.read_parquet(f"{os.environ['V0']}/train.parquet", columns=["text"]).text.map(norm))
    save("beavertails_unseen", [(r["prompt"], r["response"], int(not r["is_safe"]), "en") for r in b if norm(r["prompt"]) not in seen])

t = pd.read_csv(dl("lmsys/toxic-chat", "data/0124/toxic-chat_annotation_test.csv"))
save("toxicchat", [(r.user_input, "", int(r.toxicity), "en") for r in t.itertuples()])

o = [json.loads(x) for x in gzip.open(dl("mmathys/openai-moderation-api-evaluation", "samples-1680.jsonl.gz"), "rt")]
save("openai_mod", [(r["prompt"], "", int(any(r.get(k) == 1 for k in ["S", "H", "V", "HR", "SH", "S3", "H2", "V2"])), "en") for r in o])

x = pd.read_csv(dl("Paul/XSTest", "xstest_prompts.csv"))
save("xstest", [(r.prompt, "", int(r.label == "unsafe"), "en") for r in x.itertuples()])

s = pd.read_csv(dl("Bertievidgen/SimpleSafetyTests", "sst_test_cases.csv"))
save("simplesafety", [(r.prompt, "", 1, "en") for r in s.itertuples()])

rows = []
for code, lang in [("arb", "ar"), ("eng", "en"), ("fra", "fr"), ("hin", "hi"), ("rus", "ru"), ("spa", "es"), ("srp", "sr"), ("tgl", "tl")]:
    a = [json.loads(z) for z in open(dl("CohereForAI/aya_redteaming", f"aya_{code}.jsonl"))]
    rng.shuffle(a)
    rows += [(r["prompt"], "", 1, lang) for r in a[:300]]
save("aya_redteaming", rows)

rows = []
for lang in ["am", "ar", "de", "en", "es", "fr", "he", "hi", "it", "ja", "ru", "tt", "uk", "zh"]:
    d = pd.read_parquet(dl("textdetox/multilingual_toxicity_dataset", f"data/{lang}-00000-of-00001.parquet"))
    d = pd.concat([sample(d[d.toxic == 1], 200), sample(d[d.toxic == 0], 200)])
    rows += [(r.text, "", int(r.toxic), lang) for r in d.itertuples()]
save("textdetox", rows)
