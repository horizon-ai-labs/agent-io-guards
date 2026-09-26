"""Model card for Horizon-Labs/multilingual-zeroshot-{small,base}.  python release/zeroshot/make_zs_card.py SIZE OUT"""
import json, sys

size, out = sys.argv[1], sys.argv[2]
import os
SIZES = {"small": ("jhu-clsp/mmBERT-small", "141M"), "base": ("jhu-clsp/mmBERT-base", "308M"), "large": ("BAAI/bge-m3", "568M")}
base_model, params = SIZES[size]
others = [o for o in SIZES if o != size and os.path.exists(f"release/evals/zs_{o}.json")]
REPO = f"Horizon-Labs/multilingual-zeroshot-{size}"
me = json.load(open(f"release/evals/zs_{size}.json"))
ots = {o: json.load(open(f"release/evals/zs_{o}.json")) for o in others}
bl = json.load(open("release/evals/zs_baselines.json"))
v10 = json.load(open(f"release/evals/zs_{size}_v1.0.json")) if os.path.exists(f"release/evals/zs_{size}_v1.0.json") else None
cols = {f"**this model** ({params})": me, **{f"{o} ({SIZES[o][1]})": ots[o] for o in others},
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
        if all(v is None for v in vals):
            continue
        b = max(v for v in vals if v is not None)
        lines.append(f"| {name} | " + " | ".join("–" if v is None else ("**%.3f**" if abs(v - b) < 1e-9 else "%.3f") % v for v in vals) + " |")
    return "\n".join(lines)


multi_rows = [("MASSIVE intents (60 labels), 16 languages", lambda c: acc(c, "massive")),
              ("MTOP intents (88 labels), 6 languages", lambda c: acc(c, "mtop")),
              ("SIB-200 topics (7 labels), 16 languages §", lambda c: acc(c, "sib200"))]
lang_rows = [(NAMES[l], (lambda l: lambda c: (acc(c, "massive", l) + acc(c, "sib200", l)) / 2)(l)) for l in LANGS]
EN = [("agnews", "AG News (4) §"), ("yahoo", "Yahoo Answers (10) §"), ("banking77", "Banking77 (77)"), ("clinc", "CLINC150 (150) §"),
      ("emotion", "Emotion (6) §"), ("sst2", "SST-2 (2) §")]
en_rows = [(n, (lambda k: lambda c: acc(c, k))(k)) for k, n in EN] + \
          [("MASSIVE, English only", lambda c: acc(c, "massive", "en")), ("SIB-200, English only", lambda c: acc(c, "sib200", "en"))]
# version history: release/evals/zs_{size}_v<X>.json for earlier versions; the current one is zs_{size}.json, named by $VERSION
CUR = os.environ.get("VERSION", {"small": "v1.3", "base": "v1.3", "large": "v1.1"}[size])
PREV = sorted([f[len(f"zs_{size}_"):-5] for f in os.listdir("release/evals") if f.startswith(f"zs_{size}_v") and f.endswith(".json")])
PREV = [v for v in PREV if v < CUR]
ver_cols = ({**{v: json.load(open(f"release/evals/zs_{size}_{v}.json")) for v in PREV}, f"**{CUR} (this version)**": me}) if PREV else None
v11 = "v1.1" in PREV
ver_rows = [("MASSIVE (unseen label set)", lambda c: acc(c, "massive")),
            ("MASSIVE, native labels (15 languages)", lambda c: c.get("native_labels", {}).get("massive")),
            ("SIB-200, native labels (15 languages)", lambda c: c.get("native_labels", {}).get("sib200")), ("Banking77 (unseen label set)", lambda c: acc(c, "banking77")),
            ("MTOP (unseen label set)", lambda c: c.get("mtop", {}).get("all", {}).get("acc")),
            ("CLINC150 §", lambda c: c.get("clinc", {}).get("all", {}).get("acc")),
            ("XNLI (balanced acc.)", lambda c: c["xnli"]["all"]["bacc"]), ("SIB-200 §", lambda c: acc(c, "sib200")),
            ("AG News §", lambda c: acc(c, "agnews")), ("Yahoo Answers §", lambda c: acc(c, "yahoo")),
            ("Emotion §", lambda c: acc(c, "emotion")), ("SST-2 §", lambda c: acc(c, "sst2"))]
xnli_rows = [("XNLI test, 12 languages (balanced acc.) †", lambda c: c["xnli"]["all"]["bacc"])]

DISTILL_LINE = (("  - (v1.3) Native-language labels: label sets and templates of the non-English synthetic items translated by\n"
                 "    Qwen3.8-27B into the item's language; half of those items are trained with the native labels.\n") if CUR >= "v1.3" else "") + ("  - (v1.2) Distillation: half of the loss uses the probabilities of multilingual-zeroshot-large (568M) on the same\n"
                "    training pairs instead of the hard labels.\n" if (size == "base" and CUR >= "v1.2") else "")
_nat = json.load(open("release/evals/zs_native_labels.json")) if os.path.exists("release/evals/zs_native_labels.json") else None
NATIVE_SECTION = ""
if _nat:
    _rows = []
    for key, n in [("small", "small (141M)"), ("base", "base (308M)"), ("large", "large (568M)")]:
        if key in _nat and os.path.exists(f"release/evals/zs_{key}.json"):
            e = json.load(open(f"release/evals/zs_{key}.json"))
            _rows.append((("**" + n + "** (this model)") if key == size else n, _nat[key], e))
    for key, n in [("MoritzLaurer/bge-m3-zeroshot-v2.0-c", "bge-m3-zeroshot-v2.0-c"), ("MoritzLaurer/mDeBERTa-v3-base-xnli-multilingual-nli-2mil7", "mDeBERTa-v3-base-xnli"),
                   ("MoritzLaurer/bge-m3-zeroshot-v2.0", "bge-m3-zeroshot-v2.0 ‡")]:
        if key in _nat:
            _rows.append((n, _nat[key], bl[key]))
    def _en(e, ds):   # English-label accuracy on the same 15 non-English languages
        ls = ["de", "fr", "es", "pt", "ru", "pl", "tr", "ar", "hi", "zh", "ja", "ko", "vi", "id", "sw"]
        return sum(e[ds][l]["acc"] for l in ls) / len(ls)
    NATIVE_SECTION = "\n".join(["### Labels in the text's language", "",
        "Users often write the candidate labels and the template in the language of the text. Below, the MASSIVE and SIB-200",
        "label sets and templates were translated into each of the 15 non-English languages (by Qwen3.8-27B); accuracy is the mean",
        "over those languages, next to the same languages with English labels.", "",
        "| | MASSIVE, native labels | MASSIVE, English labels | SIB-200, native labels | SIB-200, English labels |", "|---|---|---|---|---|"] +
        [f"| {n} | {nv['massive']:.3f} | {_en(e, 'massive'):.3f} | {nv['sib200']:.3f} | {_en(e, 'sib200'):.3f} |" for n, nv, e in _rows] + ["", ""])
VDESC = {
    "v1.1": "v1.1 adds data with broad, reusable label taxonomies, so it is better on common categories (topics, emotions, sentiment,\n"
            "aspects) — the § rows, where the label names are familiar to it. On label sets it has not seen it stays within about\n"
            "±0.015 of v1.0 (slightly lower on some).",
    "v1.2": "v1.2 is distilled from the large model: same data, with half of the loss on the large model's probabilities instead of\n"
            "the hard labels. It gains most on the unseen MASSIVE label set; other rows move by about ±0.01.",
    "v1.3": "v1.3 adds native-language labels: for half of the non-English training texts, the label set and template were translated\n"
            "into the text's language (by Qwen3.8-27B). It is clearly better when labels are written in the text's language (see\n"
            "above); with English labels it is about the same or better on the unseen label sets (MASSIVE, Banking77), and the\n"
            "table shows the other rows, which move in both directions." + (" Like v1.2, it is distilled from the large model (now on\n"
            "the v1.3 data)." if size == "base" else ""),
}
_hist = [v for v in ["v1.1", "v1.2", "v1.3"] if v in PREV + [CUR] and (v != "v1.2" or size == "base")]
PIN = " or ".join(f'`revision="{v}"`' for v in PREV)
VERSION_SECTION = ("### Versions\n\n" + "\n\n".join(VDESC[v].replace(v, v + " (this version)", 1) if v == CUR else VDESC[v] for v in _hist) +
                   ("\n\n(There is no v1.2 of this size: v1.2 was the distilled base model.)" if size != "base" and CUR >= "v1.3" else "") +
                   f"\n\nTo pin an earlier model, load it with {PIN}.\n\n" + table(ver_cols, ver_rows) + "\n\n") if ver_cols else ""
ARCH_TAGS = "- xlm-roberta\n- bge-m3" if size == "large" else "- modernbert\n- mmbert"
SIZE_BULLET = ("- **Most accurate of the family**: 568M parameters (bge-m3 / XLM-RoBERTa-large backbone, MIT), 8k context. Use a GPU "
               "for throughput; ONNX included for CPU (fp32, and int8 embeddings at 1.5 GB with 99-100% top-label agreement)." if size == "large" else
               f"- **Small and fast**: {params} parameters, ModernBERT architecture (mmBERT), ONNX included for CPU and the browser.")

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
{ARCH_TAGS}
- onnx
- transformers.js
widget:
- text: "Mi pedido llegó roto y quiero que me devuelvan el dinero."
  candidate_labels: "refund request, shipping question, product praise, account problem"
  multi_class: false
- text: "Die Bundesregierung hat ein neues Klimaschutzpaket beschlossen, das den Ausbau der Windenergie beschleunigen soll."
  candidate_labels: "politics, sports, business, science and technology, entertainment"
  multi_class: false
- text: "画面はとてもきれいだけど、バッテリーが半日しか持たないのが残念です。"
  candidate_labels: "screen, battery, price, camera"
  multi_class: true
datasets:
- nyu-mll/multi_nli
- stanfordnlp/snli
- alisawuffles/WANLI
- Horizon-Labs/multilingual-zeroshot-synthetic
---

# Multilingual Zero-Shot Classifier ({size}, {params})

Classify text in 30+ languages into **any labels you choose**, with no training. Use it with the transformers
`zero-shot-classification` pipeline, like `facebook/bart-large-mnli`, but multilingual{'' if size == 'large' else ', smaller,'} and with an 8k-token
context window (fine-tuned at up to 1,024 tokens).

- **Multilingual**: the text can be in any of the languages below; labels and the hypothesis template stay in English.
- **Commercially clean**: Apache-2.0, trained only on data that allows commercial use (no XNLI, ANLI or other
  non-commercial sets). See Training.
{SIZE_BULLET}
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
not zero-shot with respect to the label names, so compare with care. For CLINC150, 31 of the 150 intent names occur in
our training data. The MASSIVE, Banking77, MTOP and XNLI label sets were not used (1-2 names each occur by chance, such
as "play music").

### Multilingual

{table(cols, multi_rows)}

Per language, mean of MASSIVE and SIB-200:

{table(cols, lang_rows)}

### English

{table(en_cols, en_rows)}

{NATIVE_SECTION}{VERSION_SECTION}### NLI

{table(cols, xnli_rows)}

† XNLI is included for reference only: xlm-roberta-large-xnli and mDeBERTa-xnli were trained on XNLI data (the
first scores 0.99, which suggests it saw the test sentences). Our models never saw XNLI.

## Limitations

- English-only models trained with more (partly non-commercial) classification data are better on English topic
  and emotion benchmarks (e.g. deberta-v3-base-zeroshot-v2.0 on Emotion and Yahoo). If you only need English, compare
  them on your data.
- bge-m3-zeroshot-v2.0 (568M, trained partly on non-commercial data) scores higher on MASSIVE, SIB-200 and MTOP.
- Zero-shot accuracy depends a lot on label wording and the template. Use descriptive labels ("request a refund"
  rather than "refund_req") and try a template that fits your task. The model links explicit wording better than
  implied categories. Example (small v1.1, multi-label, a gym review not like our training domains): "The machines are
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
{DISTILL_LINE}  - (v1.1) Generic taxonomies by Qwen3.8-27B: 24k new FineWeb / FineWeb-2 passages labelled for topic, text type,
    sentiment, audience, purpose and news section; ~130k short texts written for fixed label sets (emotion, sentiment,
    Q&A question topic, news section, customer-message topic, urgency, formality, spam) without using the label words;
    ~25k reviews in 8 domains mentioning aspects (e.g. "internet", "food") without naming them.
- The Qwen-generated classification data (short texts, taxonomies, aspects, labelled passages) is published as
  [Horizon-Labs/multilingual-zeroshot-synthetic](https://huggingface.co/datasets/Horizon-Labs/multilingual-zeroshot-synthetic).
- Not used: XNLI, ANLI, FEVER-NLI, any benchmark above.
"""
open(out, "w").write(card)
print(table(cols, multi_rows))
