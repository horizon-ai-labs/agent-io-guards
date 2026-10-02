---
license: odc-by
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
task_categories:
- text-ranking
- text-retrieval
tags:
- reranking
- distillation
- hard-negatives
- multilingual
- synthetic-queries
size_categories:
- 100K<n<1M
source_datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
pretty_name: Multilingual reranker distillation data (Qwen3-Reranker-4B scores)
configs:
- config_name: queries
  data_files:
  - split: train
    path: queries.parquet
- config_name: passages
  data_files:
  - split: train
    path: passages.parquet
---

# Multilingual reranker distillation data

The training data of [Horizon-Labs/multilingual-reranker-small](https://huggingface.co/Horizon-Labs/multilingual-reranker-small)
and [-base](https://huggingface.co/Horizon-Labs/multilingual-reranker-base): **334,400 search queries in 47 languages**.
Each query comes with 16 candidate passages (its source passage plus 15 hard negatives from dense retrieval) and a
relevance score from [Qwen3-Reranker-4B](https://huggingface.co/Qwen/Qwen3-Reranker-4B) for every (query, passage) pair, so
5.35M scored pairs. Use it to train or distil rerankers and embedding models.

**`queries`** config:

| column | |
|---|---|
| `qid` | query id |
| `query` | the query (written by Qwen3.8-27B) |
| `qtype` | `question` (natural question), `keywords` (2-6 word keyword query) or `english` (English question about a non-English passage) |
| `qlang` | language of the query (FineWeb-2 code) |
| `source_pid` | the passage the query was written for (always `candidate_pids[0]`) |
| `candidate_pids` | 16 passage ids: the source passage, then the 15 most similar passages in the same language by [bge-m3](https://huggingface.co/BAAI/bge-m3) |
| `teacher_scores` | Qwen3-Reranker-4B log-odds `log P(yes) - log P(no)` for each candidate (higher = more relevant) |

**`passages`** config: `pid`, `text` (2-6 sentences, 120-1,200 characters), `lang`, `url` (source page).

## How it was made

1. Passages: chunks of 2-6 consecutive sentences from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and
   [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (ODC-BY), about 8,000 per language (32,000 for English).
2. Queries: Qwen3.8-27B (Apache-2.0) wrote a natural question and a keyword query for each passage, in the passage's
   language, and also an English question for 15% of the non-English passages. Queries were then sampled evenly across
   languages.
3. Candidates: bge-m3 dense retrieval among passages in the same language.
4. Scores: Qwen3-Reranker-4B with its default web-search instruction (`code`: see the GitHub link below).

The near-duplicate passages that dense retrieval finds can also be relevant. The teacher scores reflect that, so treat
them as soft labels rather than assuming the source passage is the only positive.

Code: [github.com/horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards) (`rerank/`).

## Limitations

- The queries are LLM-written, not real user queries, and the relevance scores come from a model, not from people.
- The passages are web text from FineWeb-2 and inherit its content and biases. Some languages are noisier than others.

## License

ODC-BY 1.0 (the passages come from FineWeb/FineWeb-2, ODC-BY; please also respect Common Crawl's terms of use). The
queries and scores were generated with Apache-2.0 models.
