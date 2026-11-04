"""Model card for Horizon-Labs/multilingual-punctuation-truecase-{small,base}.
python release/truecase/make_tc_card.py SIZE RELEASE_DIR BOUNDARY.json|- OUT
Reads release/evals/tc_{small,base}.json (eval_tc.py, raw argmax), tc_baselines.json, punct_data_stats.json, tc_boundary.json (optional)."""
import json, os, sys
import numpy as np

size, rd, bnd_f, out = sys.argv[1:5]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-punctuation-truecase-{size}", f"Horizon-Labs/multilingual-punctuation-truecase-{other}"
PUNCT = f"Horizon-Labs/punctuation-restoration-{size}"
PARAMS = {"small": "140M", "base": "307M"}
REC = {"small": dict(lr="5e-5", epochs=2), "base": dict(lr="3e-5", epochs=1)}
first = lambda f: list(json.load(open(f)).values())[0]
me, ot = first(f"release/evals/tc_{size}.json"), first(f"release/evals/tc_{other}.json")
bl, st = json.load(open("release/evals/tc_baselines.json")), json.load(open("release/evals/punct_data_stats.json"))
bnd = json.load(open(bnd_f)) if bnd_f != "-" else None
oc = json.load(open(f"{rd}/onnx_check.json"))
X = "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"; RULE = f"rule+punctuation-restoration-{size}"
TRAIN = [{"cmn_Hani": "zho_Hans", "fas_Arab": "pes_Arab", "ekk_Latn": "est_Latn", "fil_Latn": "tgl_Latn"}.get(l, l) for l in st["languages"]]
LN = {**{l: n for l, n in st["names"].items()}, "zho_Hans": st["names"].get("cmn_Hani", "Chinese (Simplified)")}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
link = lambda m: f"[{m.split('/')[-1]}](https://huggingface.co/{m})"
SETS = [("europarl", None, "Europarl, 12 languages"), ("ted", None, "TED talks, 25 languages"), ("flores", TRAIN, "FLORES, our 88 languages"),
        ("flores", None, "FLORES, all 199 languages")]


def avg(r, s, key, langs=None):
    d = r["per_lang"][s]
    v = [d[l][key] for l in (langs or d) if l in d and d[l][key] is not None]
    return float(np.mean(v)) if v else None


def table(key):
    rows = [(f"**this model** ({PARAMS[size]})", me), (f"{link(OTHER)} ({PARAMS[other]})", ot),
            (f"{link(PUNCT)} + rule casing", bl[RULE]), (link(X), bl[X])]
    return "\n".join(["| model | " + " | ".join(n for _, _, n in SETS) + " |", "|---" * (len(SETS) + 1) + "|"] +
                     [f"| {name} | " + " | ".join(f3(avg(r, s, key, L)) for s, L, _ in SETS) + " |" for name, r in rows])


per = "\n".join(["| language | casing F1, this model | casing F1, xlm-roberta_punctuation_fullstop_truecase | punctuation F1, this model | punctuation F1, xlm-roberta_punctuation_fullstop_truecase |",
                 "|---|---|---|---|---|"] +
                [f"| {LN.get(l, l)} | {f3(me['per_lang']['flores'][l]['case_f1'])} | {f3(bl[X]['per_lang']['flores'].get(l, {}).get('case_f1'))} | "
                 f"{f3(me['per_lang']['flores'][l]['punct_f1'])} | {f3(bl[X]['per_lang']['flores'].get(l, {}).get('punct_f1'))} |"
                 for l in sorted(TRAIN, key=lambda l: LN.get(l, l)) if l in me["per_lang"]["flores"]])
q = oc["model_quantized.onnx"]

EX = {"small": "in : my name is anna i live in berlin and work for the united nations in new york\n"
                "out: My name is Anna. I live in Berlin and work for the United Nations in New York.\n"
                "in : ich heiße thomas und wohne in münchen was machst du am wochenende\n"
                "out: Ich heiße Thomas und wohne in München. Was machst du am Wochenende?",
      "base": "in : my name is anna i live in berlin and work for the united nations in new york\n"
               "out: My name is Anna, I live in Berlin and work for the United Nations in New York.\n"
               "in : ich heiße thomas und wohne in münchen was machst du am wochenende\n"
               "out: Ich heiße Thomas und wohne in München. Was machst du am Wochenende?"}
DECODE = ""
if bnd:
    b = bnd["beta"]; rv, rh = bnd["raw"], bnd["helper"]
    DECODE = f"""
### Capitals after commas

The mark after a word and the casing of the next word are separate labels, so plain argmax decoding can produce
"hello, How are you". `truecase.py` checks these boundaries: where the model predicts a comma followed by a capital, it
compares keeping both (weighted by {b}) with a full stop + capital and with a comma + lower case, using the model's
probabilities. The weight was chosen on held-out web text, not on the test sets. Effect on the test sets (mean over the 4
columns above): punctuation F1 {rv['punct']:.3f} -> {rh['punct']:.3f}, casing F1 {rv['case']:.3f} -> {rh['case']:.3f}. The tables above
are plain argmax, which is what the transformers pipeline and transformers.js give you.
"""

card = f"""---
license: apache-2.0
language:
{chr(10).join("- " + ("'" + c + "'" if c == "no" else c) for c in ["multilingual"] + st["iso"])}
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- punctuation
- punctuation-restoration
- truecasing
- capitalization
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
- text: "my name is anna i live in berlin and work for the united nations in new york"
  example_title: "English"
- text: "hola me llamo maría y vivo en madrid con mi familia"
  example_title: "Spanish"
- text: "ich heiße thomas und wohne in münchen was machst du am wochenende"
  example_title: "German"
---

# Multilingual Punctuation + Truecasing ({size}, {PARAMS[size]}): 88 languages

Restores punctuation **and capital letters** in lowercase text in one pass, for example the output of a speech-recognition (ASR)
system, subtitles or chat messages. For each word it predicts the mark that follows it (**`.` `,` `?` `:` `-`** or nothing) and
its casing (**lower**, **Capitalized** or **UPPER**). Trained on 88 languages. Built on
[mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, ONNX files for CPU and the browser (transformers.js)
included. [Try it in the browser](https://huggingface.co/spaces/Horizon-Labs/multilingual-punctuation-truecase).
{"A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as "}{link(OTHER)}.

Actual output of this model (from the release check):

```
{EX[size]}
```

- A standard token-classification model with 18 labels `<mark>|<case>` (`0|lower`, `.|Cap`, `,|UPPER`, ...), so it runs with
  the transformers pipeline and transformers.js. The included `truecase.py` turns the labels into text. It also writes
  `，。？：` in Chinese and Japanese (no word segmentation needed), `।` in Hindi and Bengali, `، ؟` in Arabic script, `;` for
  questions in Greek and `። ፣` in Amharic.
- Input is lowercased first: the model was trained on lowercase text and does not use casing you already have. If your
  text is already cased and only lacks punctuation, use {link(PUNCT)} instead.
- `onnx/model_quantized.onnx` (int8 embeddings, {q['mb']:.0f} MB) predicts the same label as fp32 for {100 * q['token_agreement']:.2f}% of tokens on
  lowercased held-out text.

## Usage

With the included helper (any language; long texts are processed in overlapping chunks):

```python
from huggingface_hub import hf_hub_download
import importlib.util
spec = importlib.util.spec_from_file_location("truecase", hf_hub_download("{REPO}", "truecase.py"))
truecase = importlib.util.module_from_spec(spec); spec.loader.exec_module(truecase)

t = truecase.Truecaser("{REPO}")
print(t("my name is anna i live in berlin and work for the united nations in new york"))
print(t("今天天气很好我们去公园散步吧你觉得怎么样"))
```

Plain transformers: the label of each word is the label of its last token. Lowercase the text first.

```python
from transformers import pipeline
tagger = pipeline("token-classification", model="{REPO}")
tagger("the nasa mission launched on monday from florida")
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const tagger = await pipeline("token-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await tagger("the nasa mission launched on monday from florida"));
```

## Evaluation

Every system gets the same input: a paragraph lowercased, with `. , ? : ; ! -` removed. The test sets were used only for
evaluation:

- **Europarl**: European Parliament proceedings, 5 consecutive sentences, 200 paragraphs in each of 12 languages.
- **TED talks** (TED2020 via OPUS): transcripts and their translations, 5 consecutive sentences, 200 paragraphs in each of 25
  languages. Spoken style.
- **FLORES-200** devtest: professionally translated news, travel and wiki text, one paragraph per article. Columns: our 88
  training languages, and all 199 FLORES languages (the extra 111 were never trained on).

Metrics are averaged over languages:

- **Casing F1**: over the words that should be Capitalized or UPPER. A word counts only with the exact class, and capitals
  predicted on words that should be lower case count as false positives. Languages are included when their test text
  contains capitals (for Chinese, Japanese, Arabic and other scripts without case, only the few Latin-script words count).
- **Casing accuracy**: all words, including the lower-case ones.
- **Punctuation F1**: as on the punctuation-restoration cards. Micro-F1 over the five marks, and a mark counts only at the
  right position.

Casing F1:

{table("case_f1")}

Punctuation F1 (lowercased input):

{table("punct_f1")}

Casing accuracy:

{table("case_acc")}

- **+ rule casing** is our punctuation-only model {link(PUNCT)}, followed by capitalizing the first word and every word after
  `.` or `?` (and English "i"). This is what you get by adding capitals to the output of a punctuation-only model. Its
  punctuation F1 is that model's own lowercased-input score on these paragraphs.
- **xlm-roberta_punctuation_fullstop_truecase** (Apache-2.0, 47 languages) was run through the `punctuators` package on
  lowercased input, with its sentence splitting turned off, and its output was re-aligned to the input words. 2.3% of words
  could not be aligned and count as "no mark, lower case", which lowers its scores somewhat. The FLORES columns include
  languages it was not trained on.
- Two training seeds per size were evaluated before release. Mean casing F1 / punctuation F1 over Europarl, TED and FLORES-199:
  small .824 / .705, base .834 / .720.
{DECODE}
<details>
<summary>Per language (FLORES, our 88 training languages)</summary>

{per}

</details>

## Training

- **Data**: the training text of punctuation-restoration v1.1: 1,168,626 windows (119M words) of
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
  web text (ODC-BY) in 88 languages, plus 49,144 synthetic spoken-style transcripts in 69 languages, written by Qwen3.8-27B
  (Apache-2.0) for this purpose. Any transcript that shared a sentence with a test set was removed.
- **Labels** come from the original text. Punctuation is normalized as for punctuation-restoration. The casing classes are:
  **Cap** (first cased letter upper case), **UPPER** (two or more cased letters, all upper case) and **lower** (everything
  else, including words in scripts without case). The model always sees the text lowercased.
- **Model**: mmBERT-{size} token classification, 18 joint labels on each word's last token, {REC[size]['epochs']} epoch{'s' if REC[size]['epochs'] > 1 else ''},
  learning rate {REC[size]['lr']} (the punctuation-restoration-{size} recipe). The checkpoint and the released seed were chosen on held-out web text
  by the mean of punctuation F1 and casing F1.
- Release criteria, fixed before training: punctuation F1 no more than .01 below {link(PUNCT)} on lowercased input; casing F1 at
  least .05 above the rule baseline on every test set and .10 above on average; and higher casing F1 than
  xlm-roberta_punctuation_fullstop_truecase on Europarl + TED. Both sizes passed.
- Code: `code/` in this repository.

## Limitations

- Only three casing classes. Mixed-case words come out wrong: `iphone` becomes `Iphone`, and `mcdonald's` becomes
  `Mcdonald's`. All-caps words are rare (0.3% of the words in the test text), so acronyms the model has not seen may stay
  lower case.
- Capitals on names depend on the model having seen the name or its context. This is weaker in low-resource languages (see
  the per-language table).
- The mark after a word and the casing of the next word are separate labels, so the model sometimes writes a comma
  followed by a capital ("Hello, How are you") where a full stop belongs or the next word should be lower case. On the
  test sets this happens for about 1 in 1,000 words or fewer, and most comma + capital predictions are right (a name
  after a comma). A decoding that re-checks these boundaries did not help on held-out text (`eval/boundary_decoding.json`),
  so `truecase.py` uses the plain predictions.
- Your casing is ignored: the input is lowercased first.
- Only `. , ? : -` are predicted. Exclamation marks, semicolons, quotes and brackets are not restored, and `:` and `-` are
  much less reliable than `.` `,` `?`. Thai, Lao, Khmer, Burmese and Tibetan are not supported.
- Trained mostly on written web text, plus synthetic spoken-style transcripts. Disfluent speech (fillers, false starts,
  repetitions) is harder.
"""
open(out, "w").write(card); print(out, len(card))
