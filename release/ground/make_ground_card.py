"""Model card for Horizon-Labs/hallucination-guard-{small,base}.  python release/ground/make_ground_card.py SIZE OUT"""
import json, sys

size, out = sys.argv[1], sys.argv[2]
SIZES = {"small": ("jhu-clsp/mmBERT-small", "141M"), "base": ("jhu-clsp/mmBERT-base", "308M")}
base_model, params = SIZES[size]
other = "base" if size == "small" else "small"
REPO = f"Horizon-Labs/hallucination-guard-{size}"
me = json.load(open(f"release/evals/ground_{size}.json"))
ot = json.load(open(f"release/evals/ground_{other}.json"))
bl = json.load(open("release/evals/ground_baselines.json"))
import os
V11 = size == "base" and os.path.exists("release/evals/ground_base_v1.0.json")
cols = {"**this model**": me, f"{other} ({SIZES[other][1]})": ot, "MiniCheck-RoBERTa-L": bl["lytang/MiniCheck-RoBERTa-Large"],
        "MiniCheck-DeBERTa-L": bl["lytang/MiniCheck-DeBERTa-v3-Large"], "HHEM-2.1-open": bl["vectara/hallucination_evaluation_model"]}
AGG = [("aggrefact_aggrefact_cnn", "AggreFact-CNN"), ("aggrefact_aggrefact_xsum", "AggreFact-XSum"), ("aggrefact_claimverify", "ClaimVerify"),
       ("aggrefact_expertqa", "ExpertQA"), ("aggrefact_factcheck_gpt", "FactCheck-GPT"), ("aggrefact_lfqa", "LFQA"),
       ("aggrefact_ragtruth", "RAGTruth †"), ("aggrefact_reveal", "Reveal"), ("aggrefact_tofueval_medias", "TofuEval-MediaS"),
       ("aggrefact_tofueval_meetb", "TofuEval-MeetB"), ("aggrefact_wice", "Wice")]
LANG = [("halueval_en", "English"), ("halueval_de", "German"), ("halueval_es", "Spanish"), ("halueval_zh", "Chinese"),
        ("halueval_ja", "Japanese"), ("halueval_ar", "Arabic"), ("halueval_hi", "Hindi")]
OTHER = [("halueval_qa", "HaluEval QA"), ("halueval_dialogue", "HaluEval dialogue"), ("halueval_summarization", "HaluEval summarization"),
         ("ragtruth_test", "RAGTruth test, response level †")]


def table(rows, extra_rows=()):
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for k, name in rows:
        vals = [c.get(k, {}).get("bacc") for c in cols.values()]
        b = max(v for v in vals if v is not None)
        lines.append(f"| {name} | " + " | ".join(("**%.3f**" if v is not None and abs(v - b) < 1e-9 else "%.3f") % v if v is not None else "–" for v in vals) + " |")
    for name, fn in extra_rows:
        vals = [fn(c) for c in cols.values()]
        b = max(vals)
        lines.append(f"| **{name}** | " + " | ".join(("**%.3f**" if abs(v - b) < 1e-9 else "%.3f") % v for v in vals) + " |")
    return "\n".join(lines)


agg_mean = lambda c: sum(c[k]["bacc"] for k, _ in AGG) / len(AGG)
nonen_mean = lambda c: sum(c[k]["bacc"] for k, _ in LANG[1:]) / (len(LANG) - 1)

def _m(c, keys): return sum(c[k]["bacc"] for k in keys) / len(keys)
def _a(c, keys): return sum(c[k]["auc"] for k in keys) / len(keys)
if V11:
    v10 = json.load(open("release/evals/ground_base_v1.0.json")); s1 = json.load(open("release/evals/ground_base_v1.1_seed1.json"))
    r1 = json.load(open("release/evals/ground_base_v1.0recipe_seed1.json"))
    AK = [k for k, _ in AGG]; NK = [k for k, _ in LANG[1:]]
    vrow = lambda n, c: f"| {n} | {_m(c, AK):.3f} | {_a(c, AK):.3f} | {_m(c, NK):.3f} | {_a(c, NK):.3f} | {c['halueval_qa']['bacc']:.3f} | {c['ragtruth_test']['bacc']:.3f} |"
    VERSION_SECTION = "\n".join([
        "### Versions", "",
        "v1.1 (this version) adds about 147k multi-sentence claim pairs (claims whose facts are spread over several sentences of",
        "the source, and the same pairs with one needed sentence removed). It is better on English claim-level fact-checking",
        "(LLM-AggreFact). On translated HaluEval it scores lower than v1.0, but a large part of that gap is training noise:",
        "retraining the v1.0 recipe with another random seed gives very different HaluEval numbers (rows below). The translated-",
        "HaluEval AUC is consistently about 0.02 lower with the v1.1 data, so part of the difference is real. Pin the previous",
        'model with `revision="v1.0"`.', "",
        "| | AggreFact bacc | AggreFact AUC | 6 languages bacc | 6 languages AUC | HaluEval QA | RAGTruth |",
        "|---|---|---|---|---|---|---|",
        vrow("v1.0 (released)", v10), vrow("v1.0 recipe, another seed", r1), vrow("**v1.1 (this version)**", me), vrow("v1.1 recipe, another seed", s1), ""])
else:
    VERSION_SECTION = ""
V11_TRAIN = ("\n  - (v1.1) About 147k multi-sentence claim pairs generated with Qwen3.8-27B (`code/ground/gen_c2d.py`): invented documents"
             " whose facts are spread over different sentences, and FineWeb / FineWeb-2 passages with claims that combine 2-3 sentences;"
             " unsupported versions remove one needed sentence. About 40% English.") if V11 else ""

card = f"""---
license: apache-2.0
language:
- multilingual
- en
- de
- fr
- es
- pt
- it
- nl
- pl
- ru
- uk
- tr
- ar
- hi
- zh
- ja
- ko
- vi
- id
- th
library_name: transformers
pipeline_tag: text-classification
base_model: {base_model}
tags:
- hallucination-detection
- groundedness
- faithfulness
- fact-checking
- rag
- llm-evaluation
- guardrails
- nli
- modernbert
- mmbert
- onnx
- transformers.js
datasets:
- wandb/RAGTruth-processed
- alisawuffles/WANLI
---

# Hallucination Guard ({size}, {params})

A small, **multilingual groundedness checker**. Given a source (retrieved documents, a transcript, a tool result) and
an AI response or claim, it predicts whether the response is **supported** by the source. Use it to flag RAG answers,
summaries and agent outputs that add, change or contradict facts.

- **Multilingual**: trained on data in 30 languages. On HaluEval translated into six languages it stays at the level
  of its English score, while English-only checkers drop (table below).
- **Long context**: 8k-token window (trained at 2k); long sources can also be chunked (see below).
- **Open**: Apache-2.0, ungated. Trained on permissively licensed data and Qwen-generated synthetic data. ONNX
  included.
- **Honest about where it loses**: on English claim-level fact-checking (LLM-AggreFact), MiniCheck and HHEM are
  better. If you only need English, compare them on your data.

Try it in the browser: [Horizon-Labs/hallucination-guard demo](https://huggingface.co/spaces/Horizon-Labs/hallucination-guard).

Part of [Agent I/O Guards](https://huggingface.co/collections/Horizon-Labs/agent-i-o-guards-6ab403c49494bc2b71ca7669)
(prompt injection, PII, groundedness). Source code: [github.com/horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards).

## Quick start

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}")
doc = "The Eiffel Tower is 330 metres tall and was completed in 1889 for the World's Fair."
clf({{"text": doc, "text_pair": "The tower was finished in 1889."}})   # SUPPORTED
clf({{"text": doc, "text_pair": "The tower was finished in 1899."}})   # UNSUPPORTED
```

Put the **source first** and the **response or claim second**. Labels: `UNSUPPORTED` (0) and `SUPPORTED` (1).
The score for `SUPPORTED` is the support probability.

For long sources, or to find *which* sentence is unsupported, split the response into sentences and score each one
against the source. For a response-level score, take the minimum over sentences:

```python
import re
def check(source, response, clf=clf):
    sents = [s for s in re.split(r"(?<=[.!?。！？])\\s+", response) if s.strip()]
    res = clf([{{"text": source, "text_pair": s}} for s in sents], truncation=True, max_length=2048)
    return [(s, r["label"], r["score"]) for s, r in zip(sents, res)]
```

## Evaluation

Balanced accuracy at a 0.5 threshold. All models were run by us with the same script
(`ground/evaluate_ground.py`). MiniCheck used context chunking with max over chunks, as its own library does. HHEM
used contexts capped at 12,000 characters, because it ran out of memory on the longest documents. † marks sets that are
**in-distribution for our models**: we trained on RAGTruth's train split, and those rows use its test split.

### LLM-AggreFact (English claim verification, 11 datasets, up to 1,000 examples each)

{table(AGG, [("Mean", agg_mean)])}

### Multilingual: HaluEval QA and dialogue, translated (400 items per language)

The items were machine-translated with Qwen3.8-27B, keeping their labels. They are disjoint from the English HaluEval
items above. Caveat: the translations come from the same model family we used to generate synthetic training data,
which may favour our model somewhat.

{table(LANG, [("Mean of the 6 non-English languages", nonen_mean)])}

{VERSION_SECTION}
### Other benchmarks

{table(OTHER)}

## Limitations

- On English claim-level fact-checking, MiniCheck-RoBERTa-L (0.749) and HHEM (0.725) beat this model
  ({agg_mean(me):.3f}) on LLM-AggreFact. Our advantage is in other languages, and in QA and dialogue grounding.
- It judges support by the given source only. It is not a world-knowledge fact checker: a true statement that the
  source doesn't contain is `UNSUPPORTED`.
- Simple arithmetic or temporal inferences are often marked `UNSUPPORTED`. For example, "opened before 2022" given a
  source that says "opened in March 2021". The small model also misses some paraphrases ("weekdays" for "Monday to
  Friday") that the base model handles.
- Summaries with many small details (HaluEval summarization) and expert long-form answers (ExpertQA) are hard for
  every model here.
- Much of the training data is synthetic (Qwen3.8-27B). The multilingual numbers come from translated data, not native
  benchmarks.

## Training

- Backbone: [{base_model}](https://huggingface.co/{base_model}) (MIT), sequence-pair classification, max length 2048,
  bf16.
- About 340k (source, response) pairs:
  - Qwen3.8-27B-generated responses over FineWeb-Edu / FineWeb-2 passages (ODC-BY) in 29 languages: supported
    answers and minimally edited unsupported variants, at both response and sentence level.
  - Qwen-generated documents in 18 genres (news, meeting transcripts, support chats, retrieved snippets, reviews,
    contracts…) with supported and unsupported claims, in 30 languages.
  - [RAGTruth](https://huggingface.co/datasets/wandb/RAGTruth-processed) train split (MIT), response level.
  - [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) (CC-BY-4.0).{V11_TRAIN}
- Not used: ANLI and other non-commercial NLI data, DocNLI (derived from non-commercial sources), and every benchmark
  above except RAGTruth's train split.
"""
open(out, "w").write(card)
print(table(AGG, [("Mean", agg_mean)]).splitlines()[-1]); print(table(LANG, [("Mean non-en", nonen_mean)]).splitlines()[-1])
