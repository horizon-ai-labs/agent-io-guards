"""Model card for Horizon-Labs/punctuation-restoration-{small,base}.
python release/punct/make_punct_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json STATS.json OUT
OURS/OTHER: eval_punct.py output with one model; BASELINES: eval_punct.py + eval_pcs.py outputs merged (Hub ids as keys)."""
import json, os, sys
import numpy as np

size, rd, ours_f, other_f, bl_f, st_f, out = sys.argv[1:8]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/punctuation-restoration-{size}", f"Horizon-Labs/punctuation-restoration-{other}"
PARAMS = {"small": "140M", "base": "307M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st = first(ours_f), json.load(open(bl_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
oc = json.load(open(f"{rd}/onnx_check.json")) if os.path.exists(f"{rd}/onnx_check.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
KREDOR = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "bul_Cyrl", "ita_Latn", "pol_Latn", "nld_Latn", "ces_Latn", "por_Latn",
          "slk_Latn", "slv_Latn"]
# FLORES codes of our 88 training languages (FineWeb-2 names differ for 4)
TRAIN = [{"cmn_Hani": "zho_Hans", "fas_Arab": "pes_Arab", "ekk_Latn": "est_Latn", "fil_Latn": "tgl_Latn"}.get(l, l) for l in st["languages"]]


def avg(r, s, mode, langs=None):
    d = r.get(s, {})
    v = [d[l][mode]["f1"] for l in (langs or d) if l in d and mode in d[l]]
    return float(np.mean(v)) if v and (langs is None or len(v) == len([l for l in langs if l in me.get(s, {})])) else None


COLS = [("flores", "cased", KREDOR), ("flores", "lower", KREDOR), ("flores", "cased", TRAIN), ("flores", "lower", TRAIN),
        ("ted", "cased", None), ("ted", "lower", None), ("europarl", "cased", None), ("europarl", "lower", None)]
row = lambda name, lic, langs, r: f"| {name} | {lic} | {langs} | " + " | ".join(f3(avg(r, s, m, L)) for s, m, L in COLS) + " |"
link = lambda m: f"[{m.split('/')[-1]}](https://huggingface.co/{m})"
BL = [("kredor/punctuate-all", "MIT", "12"), ("oliverguhr/fullstop-punctuation-multilang-large", "MIT", "4"),
      ("oliverguhr/fullstop-punctuation-multilingual-base", "MIT", "6"),
      ("1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase", "Apache-2.0", "47"),
      ("pcs_47lang", "Apache-2.0", "47")]
NAMES = {"pcs_47lang": "[punct_cap_seg_47_language](https://huggingface.co/1-800-BAD-CODE/punct_cap_seg_47_language)"}
rows = [row(f"**this model** ({PARAMS[size]})", "Apache-2.0", "88", me)] + \
       ([row(f"{link(OTHER)} ({PARAMS[other]})", "Apache-2.0", "88", ot)] if ot else []) + \
       [row(NAMES.get(m, link(m)), lic, n, bl[m]) for m, lic, n in BL if m in bl]
main = "\n".join(["| model | licence | languages | FLORES, 12 EU languages (cased) | (lowercased) | FLORES, our 88 languages (cased) | (lowercased) "
                  "| TED talks, 25 languages (cased) | (lowercased) | Europarl, 12 languages (cased) | (lowercased) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"] + rows)
X = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"; K = "kredor/punctuate-all"
LN = st["names"]
fl = sorted(TRAIN, key=lambda l: LN.get(l, l))
per = "\n".join(["| language | this model (cased) | this model (lowercased) | punctuate-all (cased) | xlm-roberta_punctuation_fullstop_truecase (lowercased) |",
                 "|---|---|---|---|---|"] +
                [f"| {LN.get(l, l)} | {f3(me['flores'][l]['cased']['f1'])} | {f3(me['flores'][l]['lower']['f1'])} | "
                 f"{f3(bl[K]['flores'].get(l, {}).get('cased', {}).get('f1'))} | {f3(bl.get(X, {}).get('flores', {}).get(l, {}).get('lower', {}).get('f1'))} |"
                 for l in fl if l in me["flores"]])
marks = "\n".join(["| mark | FLORES (cased) | FLORES (lowercased) | TED (cased) | TED (lowercased) |", "|---|---|---|---|---|"] +
                  [f"| `{m}` | " + " | ".join(f3(float(np.mean([me[s][l][mode][m] for l in me[s] if s != "flores" or l in TRAIN]))) for s, mode in
                                               [("flores", "cased"), ("flores", "lower"), ("ted", "cased"), ("ted", "lower")]) + " |"
                   for m in [".", ",", "?", ":", "-"]])
q = oc.get("model_quantized.onnx", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {q.get('mb', 0):.0f} MB) predicts the same label as fp32 for "
        f"{100 * q.get('token_agreement', 0):.2f}% of tokens on held-out text.") if q else ""
ISO = st["iso"]
rec = st["recipe"][size]
SPOKEN = (f"- **Spoken-style text** (v1.1): {rec['spoken']:,} windows of edited transcripts of everyday speech (talks, podcasts, interviews,\n"
          "  meetings, lectures, vlogs, phone calls ...) in 69 languages, written by Qwen3.8-27B (Apache-2.0) for this purpose; transcripts\n"
          "  that shared a sentence with any test set were removed.\n") if rec.get("spoken") else ""
CHANGELOG = ("\n## Changelog\n\n" + "".join(f"- **{v}**: {t}\n" for v, t in rec.get("changelog", []))) if rec.get("changelog") else ""

card = f"""---
license: apache-2.0
language:
{chr(10).join("- " + ("'" + c + "'" if c == "no" else c) for c in ["multilingual"] + ISO)}
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- punctuation
- punctuation-restoration
- asr
- speech-recognition
- transcription
- token-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "hello how are you today i hope everything is fine"
  example_title: "English"
- text: "hola cómo estás espero que todo vaya bien en el trabajo"
  example_title: "Spanish"
- text: "ich habe heute keine zeit aber morgen können wir uns treffen"
  example_title: "German"
---

# Punctuation Restoration ({size}, {PARAMS[size]}): 88 languages

Adds punctuation back to text that has none: speech-recognition (ASR) transcripts, subtitles, chat messages, voice notes,
OCR or scraped text. It predicts, after each word, one of **`.` `,` `?` `:` `-`** or nothing, in **88 languages**, with any
casing (including the all-lowercase output of many ASR systems). Built on
[mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, ONNX files for CPU and the browser (transformers.js)
included. [Try it in the browser](https://huggingface.co/spaces/Horizon-Labs/punctuation-restoration).
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- Same labels as [oliverguhr/fullstop-punctuation-multilang-large](https://huggingface.co/oliverguhr/fullstop-punctuation-multilang-large)
  and [kredor/punctuate-all](https://huggingface.co/kredor/punctuate-all) (`0 . , ? - :`), so it is a drop-in for the
  [`deepmultilingualpunctuation`](https://github.com/oliverguhr/deepmultilingualpunctuation) package.
- Chinese and Japanese work too (no word segmentation needed) with the included `punctuate.py`, which writes `，。？：` there,
  and `।` (Hindi, Bengali), `، ؟` (Arabic script), `;` for questions in Greek and `። ፣` in Amharic.
- It restores punctuation only. It does not change casing or the words.
- {onnx}

## Usage

With the included helper (any language; long texts are processed in overlapping chunks):

```python
from huggingface_hub import hf_hub_download
import importlib.util, sys
spec = importlib.util.spec_from_file_location("punctuate", hf_hub_download("{REPO}", "punctuate.py"))
punctuate = importlib.util.module_from_spec(spec); spec.loader.exec_module(punctuate)

p = punctuate.Punctuator("{REPO}")
print(p("hello how are you today i hope everything is fine"))
print(p("今天天气很好我们去公园散步吧你觉得怎么样"))
```

As a drop-in for `deepmultilingualpunctuation` (space-separated languages; that package needs transformers 4.x, because it
passes `grouped_entities`, which transformers 5 removed):

```python
from deepmultilingualpunctuation import PunctuationModel
model = PunctuationModel(model="{REPO}")
print(model.restore_punctuation("my name is anna i live in berlin what about you"))
```

Plain transformers: the label of each word is the label of its last token (`"0"` = no mark).

```python
from transformers import pipeline
tagger = pipeline("token-classification", model="{REPO}")
tagger("hello how are you today")
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const tagger = await pipeline("token-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await tagger("hello how are you today i hope everything is fine"));
```

## Evaluation

Punctuation F1: micro-averaged over the five marks (a predicted mark counts only if it is the correct mark at the correct
position), averaged over languages. Every model gets the same input: the paragraph with `. , ? : ; ! -` removed, once with its
original casing and once lowercased (as most ASR output is). Same script for every model (`code/`). The test sets were used only
for evaluation:

- **FLORES-200** devtest: professionally translated news, travel and wiki text; the consecutive sentences of one article form
  a paragraph (about 280 paragraphs per language). Columns: the 12 languages of punctuate-all, and our 88 training languages.
- **TED talks** (TED2020 via OPUS): transcripts and their translations, 5 consecutive sentences, 200 paragraphs in each of 25
  languages. Spoken style.
- **Europarl**: European Parliament proceedings, 5 consecutive sentences, 200 paragraphs in each of 12 languages.
  **punctuate-all and the fullstop models were trained on Europarl**, so this column is in-domain for them; we never trained on it.

{main}

- The 1-800-BAD-CODE models expect lowercased input and also restore casing and split sentences; they are scored only on
  lowercased input, through the `punctuators` package, by re-reading their output against the input words. Words their output
  changed beyond casing (about 2% of words for xlm-roberta_punctuation_fullstop_truecase, 11% for punct_cap_seg_47_language) could not be
  aligned and count as "no mark", which lowers their scores somewhat.
- On Europarl, punctuate-all (trained on it) is ahead of this model, most clearly on lowercased input.

Per mark (this model; mean over languages, FLORES over our 88):

{marks}

<details>
<summary>Per language (FLORES)</summary>

{per}

</details>

## Training

- **Data**: {st['n_windows']:,} text windows ({st['n_words'] / 1e6:.0f}M words) in 88 languages from
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
  (ODC-BY). The labels come from the text's own punctuation: snippets that end in a full stop, are mostly letters and are not
  lists; 1-4 snippets per window (up to 230 words), 30% cut mid-sentence. {st['lower_pct']}% of the inputs were lowercased.
{SPOKEN}- **Normalisation**: `!`, `;`, `…` and script full stops (`。 । ። ۔` ...) count as `.`; `，、،` as `,`; `？؟` (and `;` in Greek) as `?`;
  a free-standing dash as `-`. A `.` before a lowercase word (abbreviations) counts as no mark.
- **Model**: mmBERT-{size} token classification, one label per word on its last token, {st['recipe'][size]['epochs']} epoch{'s' if st['recipe'][size]['epochs'] > 1 else ''}, learning rate {st['recipe'][size]['lr']}.
  Checkpoint{' and number of epochs (2 over 1)' if size == 'small' else ''} chosen on held-out web text, not on the test sets.
- Code: `code/` in this repository.
{CHANGELOG}
## Limitations

- Only `. , ? : -` are predicted. Exclamation marks, semicolons, quotes, brackets and Spanish inverted marks are not restored.
  `:` and especially `-` (a free-standing dash) are much less reliable than `.` `,` `?` (see the per-mark table).
- Accuracy on lowercased text is lower than on cased text, because capital letters reveal sentence starts.
- {"Trained mostly on written web text, plus synthetic spoken-style transcripts" if rec.get("spoken") else "Trained on written web text"}. Disfluent speech (fillers, false starts, repetitions) is
  harder, and punctuation conventions vary by writer. Thai, Lao, Khmer, Burmese and Tibetan are not supported.
- Low-resource languages and languages outside the 88 score lower (see the per-language table).
"""
open(out, "w").write(card)
print("written", out, len(card))
