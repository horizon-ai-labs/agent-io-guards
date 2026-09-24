# Agent I/O Guards

Training, data-generation and evaluation code for small open models that protect LLM applications and agents.
All models are Apache-2.0, multilingual (mmBERT backbones) and ship ONNX / transformers.js files.

| Model | Task | Hugging Face |
|---|---|---|
| Prompt Injection Guard (small 141M, base 308M) | direct + indirect prompt-injection / jailbreak detection | [small](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small) · [base](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base) |
| PII Redactor (small 141M, base 308M) | PII and secrets detection (29 entity types, 30+ languages) | [small](https://huggingface.co/Horizon-Labs/pii-redactor-small) · [base](https://huggingface.co/Horizon-Labs/pii-redactor-base) |
| Hallucination Guard (small 141M, base 308M) | groundedness: is a response supported by its source? (multilingual) | [small](https://huggingface.co/Horizon-Labs/hallucination-guard-small) · [base](https://huggingface.co/Horizon-Labs/hallucination-guard-base) |

Related: [prompt-injection eval suite](https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite) ·
[detector leaderboard](https://huggingface.co/spaces/Horizon-Labs/prompt-injection-leaderboard) ·
browser demos for [injection](https://huggingface.co/spaces/Horizon-Labs/prompt-injection-guard) and [PII](https://huggingface.co/spaces/Horizon-Labs/pii-redactor).

## Layout

- `data/`: prompt-injection datasets from permissively licensed sources (v0), plus LLM-generated additions (v1, v2),
  deduplicated against every evaluation set.
- `gen/`: synthetic data generation with vLLM and Qwen3.8-27B (Apache-2.0).
- `train/`: sequence (pair) classification training, evaluation, the obfuscation normalizer, and ONNX export and
  quantization checks.
- `pii/`: PII span data (OpenPII, Nemotron-PII, Gretel, synthetic), BIO token-classification training, and a
  label-agnostic redaction benchmark across external datasets.
- `ground/`: groundedness data generation, training and evaluation (LLM-AggreFact, RAGTruth, HaluEval).
- `release/`: model card generators, release packaging, and helpers shipped with the models (`redact.py`,
  `llm_guard_conf.py`).

The scripts were written for a GPU cluster, so paths and job wrappers are environment-specific. They document exactly
how the published models were built and evaluated; they are not a polished library. The model cards list the data
sources and their licenses, and report results against other open models, including the benchmarks where ours are
worse.

## License

Apache-2.0 for this code. The datasets used keep their own licenses; see the model cards.
