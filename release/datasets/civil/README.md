---
license: cc0-1.0
language:
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
- sv
- da
- fi
- hu
- el
- tr
- ar
- he
- fa
- hi
- bn
- ur
- zh
- ja
- ko
- vi
- th
- id
- sw
- am
- tt
task_categories:
- text-classification
tags:
- toxicity
- civil_comments
- content-moderation
- multilingual
- machine-translated
size_categories:
- 100K<n<1M
source_datasets:
- google/civil_comments
pretty_name: Civil Comments, machine-translated into 33 languages
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
---

# Civil Comments, machine-translated into 33 languages

**Content warning: this dataset contains offensive, hateful and sexually explicit text by design.**

390,862 translations of comments from [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (Jigsaw;
CC0) into 33 languages, with the original fractional annotator scores for the 7 Detoxify labels: toxicity, severe_toxicity,
obscene, threat, insult, identity_attack and sexual_explicit. We made it to train
[Horizon-Labs/multilingual-toxicity-small](https://huggingface.co/Horizon-Labs/multilingual-toxicity-small) and
[-base](https://huggingface.co/Horizon-Labs/multilingual-toxicity-base).

| column | |
|---|---|
| `text` | the translated comment |
| `lang` | target language |
| `toxicity` ... `sexual_explicit` | the source comment's annotator scores (0-1, fraction of annotators) |
| `source_text` | the English source comment |

## How it was made

- From a 1.12M-comment sample of the Civil Comments training set (all comments with toxicity >= 0.2, plus twice as many
  others), each language got its own random sample of 12,000: half with toxicity >= 0.3, half below. So about half of the
  rows are toxic-leaning, far more than in the original.
- Translations are by Qwen3.8-27B (Apache-2.0). It was told to translate faithfully and keep insults, profanity, slurs,
  threats and intent, without softening. Outputs that failed to parse, looked like refusals, had an implausible length
  ratio or were copied unchanged were dropped (about 1.3%; there were no refusals).
- Code: [github.com/horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards) (`tox/translate.py`).

## Limitations

- These are machine translations. Slurs and insults may become milder, harsher or more literal in translation, so the
  copied English scores are not always right for the translation. Quality is lower for Amharic and Tatar.
- The labels reflect the original (mostly US/Canadian) annotators and topics, with known biases around identity terms.
- Do not use this data to generate abuse. It is meant for training and evaluating moderation systems.

## License

CC0-1.0, like the source dataset. Please also credit Jigsaw / Civil Comments (Borkan et al., 2019).
