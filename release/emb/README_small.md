---
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
base_model: jhu-clsp/mmBERT-small
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

# Multilingual Embedding (small, 141M) - bge-m3 compatible

A multilingual sentence and passage embedding model distilled from [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) into
[mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small). It outputs 1024-dimensional vectors **in bge-m3's dense vector
space**, so you can:

- use it on its own for search, RAG, clustering and similarity in about 90 languages, 4x smaller than bge-m3;
- or **query an existing bge-m3 index** with it: embed queries with this model and keep the passages you already embedded
  with bge-m3 (see the "→ bge-m3 index" row below). That gives cheap query encoding on CPU or in the browser.

A larger, more accurate version is available as [Horizon-Labs/multilingual-embedding-base](https://huggingface.co/Horizon-Labs/multilingual-embedding-base). Apache-2.0. ONNX files for CPU and the browser (transformers.js) are included.

- Vectors are L2-normalised. Use cosine similarity (= dot product). No query/passage prefixes are needed.
- `onnx/model_quantized.onnx` (int8 embeddings, 269 MB; cosine >= 0.99997 to the fp32 model on test sentences) outputs the final normalised embedding as `sentence_embedding`.

## Usage

```python
from sentence_transformers import SentenceTransformer

model = SentenceTransformer("Horizon-Labs/multilingual-embedding-small")
q = model.encode(["How tall is the Eiffel Tower?"])
p = model.encode(["La tour Eiffel mesure 330 mètres.", "The Statue of Liberty is 93 metres tall."])
print(q @ p.T)   # higher = more similar

# Querying an existing bge-m3 index: passages embedded earlier with BAAI/bge-m3 (dense, normalised)
# scores = model.encode(queries) @ bge_m3_passage_vectors.T
```

transformers.js:

```js
import { AutoTokenizer, AutoModel } from "@huggingface/transformers";
const tok = await AutoTokenizer.from_pretrained("Horizon-Labs/multilingual-embedding-small");
const model = await AutoModel.from_pretrained("Horizon-Labs/multilingual-embedding-small", { dtype: "q8" });
const { sentence_embedding } = await model(await tok(["Wie hoch ist der Eiffelturm?"], { padding: true, truncation: true }));
```

## Evaluation

nDCG@10 for ranking each query's candidates by cosine similarity, on public reranking benchmarks (MTEB versions; used
only for evaluation). MIRACL: 60 queries per language with 100 candidates each. Wikipedia: 60 queries per language with 9
candidates each. "Other" = ESCI (es, jp, us), RuBQ, T2Reranking, mMARCO-ja and AskUbuntu. Same script for every model
(`code/`); multilingual-e5 is run with its "query: "/"passage: " prefixes.

| model | MIRACL (18 languages) | Wikipedia (16 languages) | other (6 sets) | mean | note |
|---|---|---|---|---|---|
| **this model** (141M) | 0.750 | 0.935 | 0.768 | 0.818 | queries and passages embedded by this model |
| **this model** → bge-m3 index (141M) | 0.750 | 0.932 | 0.769 | 0.817 | queries by this model, passages by bge-m3 |
| [Horizon-Labs/multilingual-embedding-base](https://huggingface.co/Horizon-Labs/multilingual-embedding-base) (308M) | 0.753 | 0.934 | 0.777 | 0.821 |  |
| [BAAI/bge-m3](https://huggingface.co/BAAI/bge-m3) (568M) | 0.797 | 0.927 | 0.788 | 0.837 | the teacher |
| [intfloat/multilingual-e5-small](https://huggingface.co/intfloat/multilingual-e5-small) (118M) | 0.721 | 0.915 | 0.776 | 0.804 |  |
| [intfloat/multilingual-e5-base](https://huggingface.co/intfloat/multilingual-e5-base) (278M) | 0.736 | 0.918 | 0.775 | 0.810 |  |
| [sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2](https://huggingface.co/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2) (118M) | 0.387 | 0.837 | 0.687 | 0.637 |  |
| [sentence-transformers/all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2) (23M) | 0.172 | 0.720 | 0.621 | 0.504 | English |

- The table shows the released checkpoint. Means over two training seeds: small own .817 / against a bge-m3 index .818; base own .822 / against a bge-m3 index .816.
- Models ahead of this one (own embeddings): MIRACL: bge-m3; Wikipedia: none;
  other: bge-m3, multilingual-e5-small, multilingual-e5-base.
- This is a reranking-style test over given candidate lists, not a full-corpus retrieval benchmark.

## Training

- **Distillation**: the student (mmBERT-small + mean pooling + a linear layer to 1024 dimensions) learns to reproduce bge-m3's
  normalised dense embeddings: cosine + squared-error loss, plus a loss on the in-batch similarity matrix that keeps the
  relative geometry.
- **Text**: about 4.65 million texts: snippets of 1-8 sentences and short spans from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (ODC-BY) in 92 languages, plus about 850k search queries
  written by Qwen3.8-27B (Apache-2.0) for web passages in 47 languages.
- Code: `code/` in this repository.

## Limitations

- The student approximates bge-m3 and is weaker than it. The gap is larger for long passages, rare languages and
  specialised domains.
- Only bge-m3's **dense** vectors are reproduced, not its sparse or multi-vector (ColBERT) outputs.
- Max length 512 tokens (trained with 256). Split long documents into passages.
