# Agent I/O Guards

Training, data-generation and evaluation code for small open models that protect LLM applications and agents.
All models are Apache-2.0, multilingual (mmBERT backbones) and ship ONNX / transformers.js files.

| Model | Task | Hugging Face |
|---|---|---|
| Prompt Injection Guard (small 141M, base 308M, large 568M) | direct + indirect prompt-injection / jailbreak detection | [small](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small) · [base](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base) · [large](https://huggingface.co/Horizon-Labs/prompt-injection-guard-large) |
| PII Redactor (small 141M, base 308M) | PII and secrets detection (29 entity types, 30+ languages) | [small](https://huggingface.co/Horizon-Labs/pii-redactor-small) · [base](https://huggingface.co/Horizon-Labs/pii-redactor-base) |
| Hallucination Guard (small 141M, base 308M) | groundedness: is a response supported by its source? (multilingual) | [small](https://huggingface.co/Horizon-Labs/hallucination-guard-small) · [base](https://huggingface.co/Horizon-Labs/hallucination-guard-base) |
| Content Safety Guard (small 141M, base 308M) | unsafe prompts and LLM responses, 15 harm categories (multilingual; distilled from Qwen3Guard-Gen-8B) | [small](https://huggingface.co/Horizon-Labs/content-safety-guard-small) · [base](https://huggingface.co/Horizon-Labs/content-safety-guard-base) |
| Language Detection (small 141M) | language identification for 183 languages, robust on short strings | [small](https://huggingface.co/Horizon-Labs/language-detection-small) |
| Multilingual Sentiment (small 141M, base 308M) | negative / neutral / positive in many languages (distilled from Qwen3.8-27B) | [small](https://huggingface.co/Horizon-Labs/multilingual-sentiment-small) · [base](https://huggingface.co/Horizon-Labs/multilingual-sentiment-base) |
| Multilingual Emotions (small 141M, base 308M) | the 28 GoEmotions labels in 36 languages (go_emotions + Qwen translations) | [small](https://huggingface.co/Horizon-Labs/multilingual-emotions-small) · [base](https://huggingface.co/Horizon-Labs/multilingual-emotions-base) |
| Multilingual Reranker (small 141M, base 308M) | cross-encoder reranker for search/RAG in 40+ languages (distilled from Qwen3-Reranker-4B) | [small](https://huggingface.co/Horizon-Labs/multilingual-reranker-small) · [base](https://huggingface.co/Horizon-Labs/multilingual-reranker-base) |
| Multilingual Embedding (small 141M, base 308M) | sentence/passage embeddings in bge-m3's vector space (distilled from bge-m3) | [small](https://huggingface.co/Horizon-Labs/multilingual-embedding-small) · [base](https://huggingface.co/Horizon-Labs/multilingual-embedding-base) |
| Multilingual Toxicity (small 141M, base 308M) | Detoxify-style toxic-comment labels in 34 languages | [small](https://huggingface.co/Horizon-Labs/multilingual-toxicity-small) · [base](https://huggingface.co/Horizon-Labs/multilingual-toxicity-base) |
| Multilingual Zero-Shot Classifier (small 141M, base 308M, large 568M) | classify text in 30+ languages into any labels (NLI, `zero-shot-classification` pipeline) | [small](https://huggingface.co/Horizon-Labs/multilingual-zeroshot-small) · [base](https://huggingface.co/Horizon-Labs/multilingual-zeroshot-base) · [large](https://huggingface.co/Horizon-Labs/multilingual-zeroshot-large) |

Related: [prompt-injection eval suite](https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite) ·
[injection leaderboard](https://huggingface.co/spaces/Horizon-Labs/prompt-injection-leaderboard) ·
[content safety leaderboard](https://huggingface.co/spaces/Horizon-Labs/content-safety-leaderboard) ·
[zero-shot leaderboard](https://huggingface.co/spaces/Horizon-Labs/multilingual-zeroshot-leaderboard) ·
browser demos for [injection](https://huggingface.co/spaces/Horizon-Labs/prompt-injection-guard), [PII](https://huggingface.co/spaces/Horizon-Labs/pii-redactor),
[groundedness](https://huggingface.co/spaces/Horizon-Labs/hallucination-guard), [content safety](https://huggingface.co/spaces/Horizon-Labs/content-safety-guard), [zero-shot classification](https://huggingface.co/spaces/Horizon-Labs/multilingual-zeroshot),
[language detection](https://huggingface.co/spaces/Horizon-Labs/language-detection), [sentiment](https://huggingface.co/spaces/Horizon-Labs/multilingual-sentiment)
[emotions](https://huggingface.co/spaces/Horizon-Labs/multilingual-emotions) and [reranking](https://huggingface.co/spaces/Horizon-Labs/multilingual-reranker).

## Layout

- `data/`: prompt-injection datasets from permissively licensed sources (v0), plus LLM-generated additions (v1, v2),
  deduplicated against every evaluation set.
- `gen/`: synthetic data generation with vLLM and Qwen3.8-27B (Apache-2.0).
- `train/`: sequence (pair) classification training, evaluation, the obfuscation normalizer, and ONNX export and
  quantization checks.
- `pii/`: PII span data (OpenPII, Nemotron-PII, Gretel, synthetic), BIO token-classification training, and a
  label-agnostic redaction benchmark across external datasets.
- `ground/`: groundedness data generation, training and evaluation (LLM-AggreFact, RAGTruth, HaluEval).
- `zeroshot/`: zero-shot classifier data (Qwen-labelled passages and short texts, Qwen-translated MultiNLI/WANLI),
  training-set builder, and a multilingual benchmark (MASSIVE, SIB-200, XNLI, English classics).
- `safety/`: content-safety data (Nemotron-Safety-Guard v3, teacher-labelled prompt pools, Civil Comments), Qwen3Guard
  teacher scoring, multi-label training, and a benchmark over PolyGuard, BeaverTails, ToxicChat, OpenAI moderation,
  XSTest, SimpleSafetyTests, Aya red-teaming and textdetox.
- `lid/`: language identification data (FineWeb-2, cleaning filters, short spans, label merges), training and FLORES-200
  evaluation against GlotLID, fastText LID and papluca.
- `sentiment/`: sentiment evaluation sets (tweets, Amazon reviews, MTEB; evaluation only), FineWeb-2 snippet pool,
  Qwen3.8-27B synthetic texts and soft labels, soft-label training and baseline evaluation.
- `emotion/`: GoEmotions translation with Qwen3.8-27B, multi-label training, and evaluation on GoEmotions and BRIGHTER
  (28 languages, Ekman grouping, dev-tuned thresholds) against other emotion models.
- `rerank/`: reranker data (FineWeb-2 passages, Qwen-written queries, bge-m3 hard negatives, Qwen3-Reranker-4B scores via
  vLLM), listwise distillation, and MTEB reranking evaluation (MIRACL, Wikipedia, ESCI, RuBQ, T2, mMARCO, AskUbuntu).
- `emb/`: embedding distillation into bge-m3's vector space (FineWeb-2 text pool, bge-m3 teacher embeddings, cosine
  reranking evaluation incl. querying a bge-m3 index), sentence-transformers packaging.
- `tox/`: toxicity data (Civil Comments + Qwen translations with carried-over soft labels), multi-label training, and
  evaluation on Civil Comments and TextDetox.
- `ner/`, `gib/`: multilingual NER (Qwen span labels) and gibberish detection experiments. Neither was released: both
  missed their pre-set quality gates (see the code comments for the evaluation protocol).
- `scam/`: multilingual phishing / scam / spam detector (evals, Qwen teacher, synthetic message generator, training, release).
- `fin/`: multilingual financial sentiment experiments (not released: the models missed their pre-set FiQA quality bar).
- `punct/`: punctuation restoration (88 languages): normalisation, training windows from FineWeb-2, FLORES/TED/Europarl evals,
  training and evaluation (incl. 1-800-BAD-CODE models via punctuators); `release/punct/` card, helper `punctuate.py`, publish.
- `inj/`: Qwen3.8-27B injection judge (teacher evaluation and labelling) and the v2.2 distillation data builder.
- `release/datasets/`: cards of the published datasets (multilingual GoEmotions, multilingual Civil Comments, reranker
  distillation data).
- `release/`: model card generators, release packaging, and helpers shipped with the models (`redact.py`,
  `llm_guard_conf.py`).

The scripts were written for a GPU cluster, so paths and job wrappers are environment-specific. They document exactly
how the published models were built and evaluated; they are not a polished library. The model cards list the data
sources and their licenses, and report results against other open models, including the benchmarks where ours are
worse.

## License

Apache-2.0 for this code. The datasets used keep their own licenses; see the model cards.
