"""Model card for Horizon-Labs/multilingual-embedding-{small,base} (bge-m3-compatible students).
python release/emb/make_emb_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json BASELINES.json SEEDS_TEXT N_TEXTS OUT"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, seeds, ntexts, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-embedding-{size}", f"Horizon-Labs/multilingual-embedding-{other}"
PARAMS = {"small": "141M", "base": "308M"}
ours = json.load(open(ours_f)); oth = json.load(open(other_f)); bl = json.load(open(bl_f))
sym = next(v for k, v in ours.items() if k.startswith("ours:")); asym = next(v for k, v in ours.items() if k.startswith("asym:"))
osym = next(v for k, v in oth.items() if k.startswith("ours:"))
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: f"{x:.3f}"
BL = [("cls:BAAI/bge-m3", "568M", "MIT", "the teacher"), ("e5:intfloat/multilingual-e5-small", "118M", "MIT", ""),
      ("e5:intfloat/multilingual-e5-base", "278M", "MIT", ""), ("mean:sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", "118M", "Apache-2.0", ""),
      ("mean:sentence-transformers/all-MiniLM-L6-v2", "23M", "Apache-2.0", "English")]
link = lambda m: f"[{m.split(':')[-1]}](https://huggingface.co/{m.split(':')[-1]})"
row = lambda name, p, r, note: f"| {name} ({p}) | {f3(r['_miracl'])} | {f3(r['_wiki'])} | {f3(r['_other'])} | {f3(r['_all'])} | {note} |"
rows = [row("**this model**", PARAMS[size], sym, "queries and passages embedded by this model"),
        row("**this model** → bge-m3 index", PARAMS[size], asym, "queries by this model, passages by bge-m3"),
        row(link(OTHER), PARAMS[other], osym, "")] + [row(link(m), p, bl[m], note) for m, p, lic, note in BL if m in bl]
main = "\n".join(["| model | MIRACL (18 languages) | Wikipedia (16 languages) | other (6 sets) | mean | note |", "|---|---|---|---|---|---|"] + rows)
ahead = {f: [m.split("/")[-1] for m, *_ in BL if m in bl and bl[m][f] > sym[f]] for f in ["_miracl", "_wiki", "_other"]}
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB; cosine >= {gq.get('min_cos', 0):.4f} to the fp32 model on test "
        f"sentences) outputs the final normalised embedding as `sentence_embedding`.") if gq else ""

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
- ro
- bg
- sv
- da
- 'no'
- fi
- hu
- el
- tr
- ar
- he
- fa
- ur
- hi
- bn
- ta
- te
- mr
- zh
- ja
- ko
- vi
- th
- id
- ms
- sw
- af
- am
- hy
- az
- eu
- ka
- kk
- km
library_name: sentence-transformers
pipeline_tag: sentence-similarity
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- sentence-transformers
- feature-extraction
- sentence-similarity
- embeddings
- retrieval
- multilingual
- bge-m3
- distillation
- modernbert
- mmbert
- onnx
- transformers.js
---

# Multilingual Embedding ({size}, {PARAMS[size]}) - bge-m3 compatible

A multilingual sentence and passage embedding model distilled from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) into
[mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}). It outputs 1024-dimensional vectors **in bge-m3's dense vector
space**, so you can:

- use it on its own for search, RAG, clustering and similarity in about 90 languages, {"4x" if size == "small" else "2x"} smaller than bge-m3;
- or **query an existing bge-m3 index** with it: embed queries with this model and keep the passages you already embedded
  with bge-m3 (see the "→ bge-m3 index" row below). That gives cheap query encoding on CPU or in the browser.

{"A larger, more accurate version is available as" if size == "small" else "A smaller, faster version is available as"} {link(OTHER)}.
Apache-2.0. ONNX files for CPU and the browser (transformers.js) are included.

- Vectors are L2-normalised. Use cosine similarity (= dot product). No query/passage prefixes are needed.
- {onnx}

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("{REPO}")
q = model.encode(["How tall is the Eiffel Tower?"])
p = model.encode(["La tour Eiffel mesure 330 mètres.", "The Statue of Liberty is 93 metres tall."])
print(q @ p.T)   # higher = more similar

# Querying an existing bge-m3 index: passages embedded earlier with BAAI/bge-m3 (dense, normalised)
# scores = model.encode(queries) @ bge_m3_passage_vectors.T
```

transformers.js:

```js
import {{ AutoTokenizer, AutoModel }} from "@huggingface/transformers";
const tok = await AutoTokenizer.from_pretrained("{REPO}");
const model = await AutoModel.from_pretrained("{REPO}", {{ dtype: "q8" }});
const {{ sentence_embedding }} = await model(await tok(["Wie hoch ist der Eiffelturm?"], {{ padding: true, truncation: true }}));
```

## Evaluation

nDCG@10 for ranking each query's candidates by cosine similarity, on public reranking benchmarks (MTEB versions; used
only for evaluation). MIRACL: 60 queries per language with 100 candidates each. Wikipedia: 60 queries per language with 9
candidates each. "Other" = ESCI (es, jp, us), RuBQ, T2Reranking, mMARCO-ja and AskUbuntu. Same script for every model
(`code/`); multilingual-e5 is run with its "query: "/"passage: " prefixes.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- Models ahead of this one (own embeddings): MIRACL: {", ".join(ahead["_miracl"]) or "none"}; Wikipedia: {", ".join(ahead["_wiki"]) or "none"};
  other: {", ".join(ahead["_other"]) or "none"}.
- This is a reranking-style test over given candidate lists, not a full-corpus retrieval benchmark.

## Training

- **Distillation**: the student (mmBERT-{size} + mean pooling + a linear layer to 1024 dimensions) learns to reproduce bge-m3's
  normalised dense embeddings: cosine + squared-error loss, plus a loss on the in-batch similarity matrix that keeps the
  relative geometry.
- **Text**: about {ntexts} texts: snippets of 1-8 sentences and short spans from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (ODC-BY) in 92 languages, plus about 850k search queries
  written by Qwen3.8-27B (Apache-2.0) for web passages in 47 languages.
- Code: `code/` in this repository.

## Limitations

- The student approximates bge-m3 and is weaker than it. The gap is larger for long passages, rare languages and
  specialised domains.
- Only bge-m3's **dense** vectors are reproduced, not its sparse or multi-vector (ColBERT) outputs.
- Max length 512 tokens (trained with 256). Split long documents into passages.
"""
open(out, "w").write(card)
print("written", out, len(card))
