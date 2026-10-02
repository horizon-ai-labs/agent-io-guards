---
license: apache-2.0
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
- 'no'
- fi
- hu
- el
- tr
- ar
- he
- fa
- hi
- mr
- bn
- ur
- zh
- ja
- ko
- vi
- th
- id
- sw
- af
- tt
- ha
task_categories:
- text-classification
task_ids:
- multi-label-classification
tags:
- emotion
- go_emotions
- multilingual
- machine-translated
size_categories:
- 100K<n<1M
source_datasets:
- google-research-datasets/go_emotions
pretty_name: GoEmotions, machine-translated into 35 languages
configs:
- config_name: default
  data_files:
  - split: train
    path: train.parquet
---

# GoEmotions, machine-translated into 35 languages

415,033 translations of the [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions) training
set (English Reddit comments with 28 emotion labels from human annotators) into 35 languages. The original human labels
are kept. We made it to train [Horizon-Labs/multilingual-emotions-small](https://huggingface.co/Horizon-Labs/multilingual-emotions-small)
and [-base](https://huggingface.co/Horizon-Labs/multilingual-emotions-base).

| column | |
|---|---|
| `text` | the translated comment |
| `labels` | GoEmotions labels of the source comment (list of label names, from the "simplified" config) |
| `lang` | target language |
| `source_text` | the English source comment |
| `source_index` | row index in the GoEmotions "simplified" train split |

## How it was made

- Each language got its own random sample of 12,000 of the 43,410 training comments, so together the languages cover the
  whole training set.
- Translations are by Qwen3.8-27B (Apache-2.0), asked to keep meaning, tone, emotion, slang, swearing and emojis, and to keep
  placeholders such as `[NAME]` and `[RELIGION]` unchanged.
- We dropped outputs that failed to parse, looked like refusals, had an implausible length ratio or were copied unchanged
  (about 1.2%).
- Code: [github.com/horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards) (`emotion/translate_go.py`).

## Limitations

- These are machine translations, not native text. Nuance, idioms and culture-specific emotion cues can shift, so the
  copied labels are not always right for the translation. Quality is lower for lower-resource languages (e.g. Hausa, Tatar).
- The source is English Reddit, with its topics and register. Some comments are offensive.
- Labels are subjective, as in the original dataset.

## License and attribution

Apache-2.0, like GoEmotions. Please cite the original dataset:
Demszky et al., "GoEmotions: A Dataset of Fine-Grained Emotions", ACL 2020.
