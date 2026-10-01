"""Model card for Horizon-Labs/multilingual-reranker-{small,base}.
python release/rerank/make_rr_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json BASELINES.json SEEDS_TEXT OUT"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, seeds, out = sys.argv[1:8]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-reranker-{size}", f"Horizon-Labs/multilingual-reranker-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl = first(ours_f), json.load(open(bl_f))
ot = first(other_f) if other_f != "-" else None   # "-": the other size is not released (yet)
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
BL = [("cross-encoder/ms-marco-MiniLM-L6-v2", "22M", "Apache-2.0", "English"), ("BAAI/bge-reranker-base", "278M", "MIT", "Chinese/English"),
      ("Alibaba-NLP/gte-reranker-modernbert-base", "149M", "Apache-2.0", "English"),
      ("BAAI/bge-reranker-v2-m3", "568M", "Apache-2.0", "multilingual; trained on MIRACL train"),
      ("qwen3rr:Qwen/Qwen3-Reranker-0.6B", "596M", "Apache-2.0", "multilingual LLM reranker"),
      ("qwen3rr:Qwen/Qwen3-Reranker-4B", "4B", "Apache-2.0", "multilingual LLM reranker; **our teacher**")]
link = lambda m: f"[{m.split(':')[-1]}](https://huggingface.co/{m.split(':')[-1]})"
row = lambda name, p, lic, r, note: f"| {name} ({p}) | {lic} | {f3(r['_miracl'])} | {f3(r['_wiki'])} | {f3(r['_other'])} | {f3(r['_all'])} | {note} |"
rows = [row("**this model**", PARAMS[size], "Apache-2.0", me, "multilingual")] + \
       ([row(link(OTHER), PARAMS[other], "Apache-2.0", ot, "multilingual")] if ot else []) + [row(link(m), p, lic, bl[m], note) for m, p, lic, note in BL]
main = "\n".join(["| model | licence | MIRACL (18 languages) | Wikipedia (16 languages) | other (6 sets) | mean | note |", "|---|---|---|---|---|---|---|"] + rows)
OTH = {"esci_es": "ESCI product search, Spanish", "esci_jp": "ESCI product search, Japanese", "esci_us": "ESCI product search, English",
       "rubq": "RuBQ (Russian)", "t2": "T2Reranking (Chinese)", "mmarco": "mMARCO (Japanese)", "askubuntu": "AskUbuntu duplicate questions (English)"}
V2, Q4 = bl["BAAI/bge-reranker-v2-m3"], bl["qwen3rr:Qwen/Qwen3-Reranker-4B"]
sets = sorted(k for k in me if not k.startswith("_"))
per = "\n".join(["| set | this model | bge-reranker-v2-m3 | Qwen3-Reranker-4B (teacher) |", "|---|---|---|---|"] +
                [f"| {OTH.get(k, k.replace('miracl_', 'MIRACL ').replace('wiki_', 'Wikipedia '))} | {f3(me[k]['ndcg10'])} | {f3(V2[k]['ndcg10'])} | {f3(Q4[k]['ndcg10'])} |" for k in sets])
ahead = {f: [m.split("/")[-1] for m, *_ in BL if bl[m][f] > me[f]] for f in ["_miracl", "_wiki", "_other"]}
speed = f"{me['_pairs_per_s']:.0f}"
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB): scores differ from fp32 by {gq.get('mean_diff', 0):.3f} on average "
        f"(sigmoid scale) over {qs.get('n', 0)} benchmark pairs.") if gq else ""

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
- cs
- sk
- ro
- bg
- hr
- sr
- sl
- sv
- da
- 'no'
- fi
- et
- lv
- lt
- hu
- el
- tr
- ar
- he
- fa
- ur
- hi
- mr
- bn
- ta
- te
- zh
- ja
- ko
- vi
- th
- id
- ms
- tl
- sw
- yo
- ca
library_name: transformers
pipeline_tag: text-ranking
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- reranker
- cross-encoder
- text-ranking
- sentence-transformers
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
---

# Multilingual Reranker ({size}, {PARAMS[size]})

A cross-encoder reranker for search and RAG in 40+ languages, built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}).
Give it a query and candidate passages, for example the top 20-100 hits of a vector or keyword search. It scores each
pair, so you can re-sort the candidates and keep the best ones for your LLM. Queries and passages can be in different
languages. It is distilled from [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) into a model
{"about 28x" if size == "small" else "about 13x"} smaller. Apache-2.0, trained on openly licensed web text. ONNX files
for CPU and the browser (transformers.js) are included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ". ") if ot else ""}[Try it in the browser](https://huggingface.co/spaces/Horizon-Labs/multilingual-reranker).

- One output logit per (query, passage) pair: higher = more relevant. `sigmoid(logit)` gives a 0-1 relevance score.
- Max length: trained with 384 tokens per pair. Longer passages are truncated; split long documents into chunks.
- {onnx}

## Usage

sentence-transformers:

```python
from sentence_transformers import CrossEncoder

model = CrossEncoder("{REPO}")
query = "How tall is the Eiffel Tower?"
passages = ["The Eiffel Tower is 330 metres tall.", "La tour Eiffel a été construite pour l'Exposition universelle de 1889.",
            "The Statue of Liberty is 93 metres tall."]
print(model.rank(query, passages))   # [{{'corpus_id': 0, 'score': ...}}, ...]
```

transformers:

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("{REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{REPO}").eval()
enc = tok([query] * len(passages), passages, padding=True, truncation=True, max_length=512, return_tensors="pt")
with torch.no_grad():
    scores = model(**enc).logits[:, 0].sigmoid()
```

transformers.js:

```js
import {{ AutoTokenizer, AutoModelForSequenceClassification }} from "@huggingface/transformers";
const tok = await AutoTokenizer.from_pretrained("{REPO}");
const model = await AutoModelForSequenceClassification.from_pretrained("{REPO}", {{ dtype: "q8" }});
const enc = await tok(new Array(passages.length).fill(query), {{ text_pair: passages, padding: true, truncation: true }});
const {{ logits }} = await model(enc);
```

## Evaluation

nDCG@10 for reranking the candidate lists of public reranking benchmarks (the MTEB versions). The benchmarks were used only
for evaluation, never for training or model selection. Every model sees the same queries and candidates, through the same
script (`code/`). MIRACL: 60 queries per language with 100 candidates each. Wikipedia (WikipediaRerankingMultilingual):
60 queries per language with 9 candidates each. "Other" = ESCI (es, jp, us), RuBQ, T2Reranking, mMARCO-ja and AskUbuntu.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- Models ahead of this one: MIRACL: {", ".join(ahead["_miracl"]) or "none"}; Wikipedia: {", ".join(ahead["_wiki"]) or "none"};
  other: {", ".join(ahead["_other"]) or "none"}. bge-reranker-v2-m3 was trained on MIRACL's training set; we were not.

Per set (nDCG@10):

{per}

## Training

- **Passages**: about 400k passages of 2-6 sentences from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (ODC-BY) in 47 languages.
- **Queries**: Qwen3.8-27B (Apache-2.0) wrote a natural question and a keyword query for each passage, in the passage's
  language, plus English questions for 15% of the non-English passages (cross-lingual search).
- **Scale**: 334,400 queries with 16 candidates each (5.35M teacher-scored pairs), sampled evenly across languages.
- **Schedule**: {"1 epoch, learning rate 5e-5" if size == "small" else "2 epochs, learning rate 3e-5 (2 epochs were chosen over 1 by validation loss, not by the benchmarks)"}, 16 queries x 16 candidates per step, max 384 tokens per pair.
- **Candidates**: for each query, the 15 most similar passages in the same language by [bge-m3](https://huggingface.co/BAAI/bge-m3)
  dense retrieval (hard negatives) plus the source passage.
- **Labels**: [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) (Apache-2.0) scored every (query,
  candidate) pair. The student learns the teacher's ranking with a listwise KL loss over each query's 16 candidates, plus a
  pointwise loss on the teacher's relevance probability. The teacher's scores are soft labels, so the near-duplicate
  passages that dense retrieval finds are not wrongly treated as negatives.
- Code: `code/` in this repository.

## Limitations

- The model inherits the teacher's judgement, including its mistakes. It is weaker than the teacher, especially on
  long or technical passages.
- Training queries are LLM-written questions and keyword queries over web text. Very domain-specific search (legal,
  medical, code) and conversational queries are less covered.
- Pairs are truncated at the max length. Rerank passage-sized chunks, not whole documents.
"""
open(out, "w").write(card)
print("written", out, len(card))
