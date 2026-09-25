"""Model card for Horizon-Labs/multilingual-zeroshot-{small,base}.  python release/zeroshot/make_zs_card.py SIZE OUT"""
import json, sys

size, out = sys.argv[1], sys.argv[2]
SIZES = {"small": ("jhu-clsp/mmBERT-small", "141M"), "base": ("jhu-clsp/mmBERT-base", "308M")}
base_model, params = SIZES[size]
other = "base" if size == "small" else "small"
REPO = f"Horizon-Labs/multilingual-zeroshot-{size}"
me = json.load(open(f"release/evals/zs_{size}.json"))
ot = json.load(open(f"release/evals/zs_{other}.json"))
bl = json.load(open("release/evals/zs_baselines.json"))
v10 = json.load(open(f"release/evals/zs_{size}_v1.0.json"))
cols = {f"**this model** ({params})": me, f"{other} ({SIZES[other][1]})": ot,
        "bge-m3-zeroshot-v2.0-c (568M)": bl["MoritzLaurer/bge-m3-zeroshot-v2.0-c"],
        "mDeBERTa-v3-base-xnli (278M)": bl["MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7"],
        "xlm-roberta-large-xnli (560M)": bl["joeddav/xlm-roberta-large-xnli"],
        "bge-m3-zeroshot-v2.0 (568M) ‡": bl["MoritzLaurer/bge-m3-zeroshot-v2.0"]}
en_cols = dict(cols)
en_cols.update({"bart-large-mnli (407M)": bl["facebook/bart-large-mnli"],
                "deberta-v3-base-zeroshot-v2.0 (184M) ‡": bl["MoritzLaurer/deberta-v3-base-zeroshot-v2.0"]})
LANGS = ["en", "de", "fr", "es", "pt", "ru", "pl", "tr", "ar", "hi", "zh", "ja", "ko", "vi", "id", "sw"]
NAMES = dict(en="English", de="German", fr="French", es="Spanish", pt="Portuguese", ru="Russian", pl="Polish", tr="Turkish",
             ar="Arabic", hi="Hindi", zh="Chinese", ja="Japanese", ko="Korean", vi="Vietnamese", id="Indonesian", sw="Swahili")


def acc(c, ds, lang="all"):
    return c[ds][lang]["acc"]


def table(cs, rows):
    lines = ["| | " + " | ".join(cs) + " |", "|---|" + "---|" * len(cs)]
    for name, fn in rows:
        vals = [fn(c) for c in cs.values()]
        b = max(vals)
        lines.append(f"| {name} | " + " | ".join(("**%.3f**" if abs(v - b) < 1e-9 else "%.3f") % v for v in vals) + " |")
    return "\n".join(lines)


multi_rows = [("MASSIVE intents (60 labels), 16 languages", lambda c: acc(c, "massive")),
              ("SIB-200 topics (7 labels), 16 languages §", lambda c: acc(c, "sib200"))]
lang_rows = [(NAMES[l], (lambda l: lambda c: (acc(c, "massive", l) + acc(c, "sib200", l)) / 2)(l)) for l in LANGS]
EN = [("agnews", "AG News (4) §"), ("yahoo", "Yahoo Answers (10) §"), ("banking77", "Banking77 (77)"),
      ("emotion", "Emotion (6) §"), ("sst2", "SST-2 (2) §")]
en_rows = [(n, (lambda k: lambda c: acc(c, k))(k)) for k, n in EN] + \
          [("MASSIVE, English only", lambda c: acc(c, "massive", "en")), ("SIB-200, English only", lambda c: acc(c, "sib200", "en"))]
ver_cols = {"v1.0": v10, "**v1.1 (this version)**": me}
ver_rows = [("MASSIVE (unseen label set)", lambda c: acc(c, "massive")), ("Banking77 (unseen label set)", lambda c: acc(c, "banking77")),
            ("XNLI (balanced acc.)", lambda c: c["xnli"]["all"]["bacc"]), ("SIB-200 §", lambda c: acc(c, "sib200")),
            ("AG News §", lambda c: acc(c, "agnews")), ("Yahoo Answers §", lambda c: acc(c, "yahoo")),
            ("Emotion §", lambda c: acc(c, "emotion")), ("SST-2 §", lambda c: acc(c, "sst2"))]
xnli_rows = [("XNLI test, 12 languages (balanced acc.) †", lambda c: c["xnli"]["all"]["bacc"])]

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
- tr
- ar
- fa
- he
- hi
- bn
- zh
- ja
- ko
- vi
- id
- th
- sw
- el
library_name: transformers
pipeline_tag: zero-shot-classification
base_model: {base_model}
tags:
- zero-shot-classification
- zero-shot
- nli
- natural-language-inference
- text-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
datasets:
- nyu-mll/multi_nli
- stanfordnlp/snli
- alisawuffles/WANLI
---

# Multilingual Zero-Shot Classifier ({size}, {params})

Classify text in 30+ languages into **any labels you choose**, with no training. Use it with the transformers
`zero-shot-classification` pipeline, like `facebook/bart-large-mnli`, but multilingual, smaller, and with an 8k-token
context window (fine-tuned at up to 1,024 tokens).

- **Multilingual**: the text can be in any of the languages below; labels and the hypothesis template stay in English.
- **Commercially clean**: Apache-2.0, trained only on data that allows commercial use (no XNLI, ANLI or other
  non-commercial sets). See Training.
- **Small and fast**: {params} parameters, ModernBERT architecture (mmBERT), ONNX included for CPU and the browser.
- **Honest numbers**: all models below were run by us with the same script and templates.

Try it in the browser: [Horizon-Labs/multilingual-zeroshot demo](https://huggingface.co/spaces/Horizon-Labs/multilingual-zeroshot).

Part of Horizon Labs' open models ([collection](https://huggingface.co/collections/Horizon-Labs/agent-i-o-guards-6ab403c49494bc2b71ca7669)).
Source code: [github.com/horizon-ai-labs/agent-io-guards](https://github.com/horizon-ai-labs/agent-io-guards).

## Quick start

```python
from transformers import pipeline

clf = pipeline("zero-shot-classification", model="{REPO}")
clf("Mi pedido llegó roto y quiero que me devuelvan el dinero.",
    candidate_labels=["refund request", "shipping question", "product praise", "account problem"])
# {{'labels': ['refund request', ...], 'scores': [...]}}

# several labels can apply at once
clf("The camera is great but the battery dies by noon.", ["camera", "battery", "screen", "price"], multi_label=True)

# a task-specific template often helps
clf("¿Me pones una alarma a las siete?", ["set an alarm", "play music", "weather"], hypothesis_template="The user wants to {{}}.")
```

Labels: `not_entailment` (0) and `entailment` (1). For each candidate label the model scores whether the text entails
"This example is {{label}}." (or your `hypothesis_template`). Any NLI-style use works too: pass `text` and `text_pair`
to a `text-classification` pipeline.

## Evaluation

Accuracy, single-label (`multi_label=False`: the label with the highest entailment score wins). English templates and
labels for every language; the same template for every model (e.g. "This text is about {{}}." for SIB-200). No model saw
these datasets' training splits, except where marked. ‡ = trained partly on data with non-commercial licenses (their
`-c` variants are the commercially usable ones). Script: `zeroshot/evaluate_zs.py`.

§ = **label names seen in our synthetic training data**. Since v1.1 our training data includes generic label
taxonomies (topics, news sections, Q&A question topics, emotions, sentiment) whose label names overlap these benchmarks'
label sets; for Yahoo Answers and AG News almost exactly. No benchmark texts were used, but on these rows our models are
not zero-shot with respect to the label names, so compare with care. MASSIVE, Banking77 and XNLI label sets were not used.

### Multilingual

{table(cols, multi_rows)}

Per language, mean of MASSIVE and SIB-200:

{table(cols, lang_rows)}

### English

{table(en_cols, en_rows)}

### v1.0 → v1.1

v1.1 adds data with broad, reusable label taxonomies, so it is better on common categories (topics, emotions,
sentiment, aspects) — the § rows, where the label names are familiar to it. On label sets it has not seen it stays
within about ±0.015 of v1.0 (slightly lower on some). To pin the previous model, load it with `revision="v1.0"`.

{table(ver_cols, ver_rows)}

### NLI

{table(cols, xnli_rows)}

† XNLI is included for reference only: xlm-roberta-large-xnli and mDeBERTa-xnli were trained on XNLI data (the
first scores 0.99, which suggests it saw the test sentences). Our models never saw XNLI.

## Limitations

- English-only models trained with more (partly non-commercial) classification data are better on English topic
  and emotion benchmarks (e.g. deberta-v3-base-zeroshot-v2.0 on Emotion and Yahoo). If you only need English, compare
  them on your data.
- bge-m3-zeroshot-v2.0 (568M, trained partly on non-commercial data) scores higher on MASSIVE and SIB-200.
- Zero-shot accuracy depends a lot on label wording and the template. Use descriptive labels ("request a refund"
  rather than "refund_req") and try a template that fits your task. The model links explicit wording better than
  implied categories. Example (small model, multi-label, a gym review not like our training domains): "The machines are
  always taken after 5pm and half the treadmills are broken, but the coaches really know their stuff. For 60 euros a
  month I expected cleaner showers." gives equipment 0.99, trainers 0.97, membership cost 0.82, but hygiene only 0.28
  and crowding 0.03 (v1.0: trainers 0.56, membership cost 0.58).
- Broad labels (e.g. "world news", "education") tend to win over specific ones. Emotions close in meaning (joy / love
  / surprise) are often confused.
- With `multi_label=True`, scores are independent; tune the threshold on a few examples of your own.
- Lower-resource languages (e.g. Swahili) score clearly lower than high-resource ones.
- Much of the training data is synthetic (Qwen3.8-27B) or machine-translated.

## Training

- Backbone: [{base_model}](https://huggingface.co/{base_model}) (MIT), sequence-pair classification, bf16, max length 1024.
- Data (label = does the text entail the hypothesis):
  - English NLI: [MultiNLI](https://huggingface.co/datasets/nyu-mll/multi_nli) (OANC and CC-BY-SA-3.0 parts),
    [SNLI](https://huggingface.co/datasets/stanfordnlp/snli) (CC-BY-SA-4.0), [WANLI](https://huggingface.co/datasets/alisawuffles/WANLI) (CC-BY-4.0).
  - 120k MultiNLI/WANLI pairs machine-translated by Qwen3.8-27B into 24 languages, with native and English hypotheses.
  - Synthetic zero-shot tasks by Qwen3.8-27B: FineWeb-Edu / FineWeb-2 passages (ODC-BY) labelled by topic, genre,
    audience, tone and purpose with near-miss wrong labels, and ~90k short texts (requests, reviews, tickets, posts,
    headlines) over 26 task types, 32 domains and 33 languages, each with an invented label set and hypothesis template.
  - (v1.1) Generic taxonomies by Qwen3.8-27B: 24k new FineWeb / FineWeb-2 passages labelled for topic, text type,
    sentiment, audience, purpose and news section; ~130k short texts written for fixed label sets (emotion, sentiment,
    Q&A question topic, news section, customer-message topic, urgency, formality, spam) without using the label words;
    ~25k reviews in 8 domains mentioning aspects (e.g. "internet", "food") without naming them.
- Not used: XNLI, ANLI, FEVER-NLI, any benchmark above.
"""
open(out, "w").write(card)
print(table(cols, multi_rows))
