"""Model card for Horizon-Labs/language-detection-small.
python release/lid/make_lid_card.py RELEASE_DIR OURS_EVAL.json BASELINES_EVAL.json STATS_V0.json OUT [MERGES.json]
(OURS_EVAL and BASELINES_EVAL may be the same file: the first entry is taken as ours.)"""
import json, os, sys

rd, ours_f, bl_f, stats_f, out = sys.argv[1:6]
REPO = "Horizon-Labs/language-detection-small"
me = list(json.load(open(ours_f)).values())[0]
bl = json.load(open(bl_f))
st = json.load(open(stats_f))
labels = json.load(open(f"{rd}/labels.json"))
merges = json.load(open(sys.argv[6])) if len(sys.argv) > 6 else {"merge": {}, "drop": []}
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
G, F, P = bl["fasttext:cis-lmu/glotlid"], bl["fasttext:facebook/fasttext-language-identification"], bl["papluca:papluca/xlm-roberta-base-language-detection"]
isos = sorted({l.split("_")[0] for l in labels})
f3 = lambda x: f"{x:.3f}"

main = "\n".join([
    "| | **this model** (141M) | GlotLID (fastText, 1.7 GB) | fastText LID-218 (NLLB, CC-BY-NC) |", "|---|---|---|---|",
    f"| FLORES-200 devtest, full sentences: accuracy | **{f3(me['flores_devtest']['acc'])}** | {f3(G['flores_devtest']['acc'])} | {f3(F['flores_devtest']['acc'])} |",
    f"| FLORES-200 devtest, full sentences: macro-F1 | **{f3(me['flores_devtest']['macro_f1'])}** | {f3(G['flores_devtest']['macro_f1'])} | {f3(F['flores_devtest']['macro_f1'])} |",
    f"| First 20-40 characters: accuracy | **{f3(me['flores_short']['acc'])}** | {f3(G['flores_short']['acc'])} | {f3(F['flores_short']['acc'])} |",
    f"| First 20-40 characters: macro-F1 | **{f3(me['flores_short']['macro_f1'])}** | {f3(G['flores_short']['macro_f1'])} | {f3(F['flores_short']['macro_f1'])} |"])
pap = "\n".join([
    f"| papluca's 20 languages | **this model**, all {len(labels)} labels | **this model**, restricted to the 20 | papluca/xlm-roberta-base-language-detection (278M) |", "|---|---|---|---|",
    f"| full sentences (accuracy) | {f3(me['flores_devtest_papluca20']['acc'])} | **{f3(me['flores_devtest_papluca20_restricted']['acc'])}** | {f3(P['flores_devtest']['acc'])} |",
    f"| first 20-40 characters (accuracy) | {f3(me['flores_short_papluca20']['acc'])} | **{f3(me['flores_short_papluca20_restricted']['acc'])}** | {f3(P['flores_short']['acc'])} |"])
pl, gl = me["flores_devtest"]["per_lang"], G["flores_devtest"]["per_lang"]
weak = sorted([(l, a) for l, a in pl.items() if a < 0.9], key=lambda x: x[1])
weak_t = "\n".join(["| language code | this model | GlotLID |", "|---|---|---|"] + [f"| `{l}` | {a:.2f} | {gl.get(l, float('nan')):.2f} |" for l, a in weak])
dropped = sorted(set(st.get("dropped_no_fineweb", [])) | set(st.get("dropped_too_few", [])) | set(merges.get("drop", [])))
arab = sorted(k for k, v in merges.get("merge", {}).items() if v == "arb_Arab")
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB) picks the same language as fp32 on "
        f"{100 * gq.get('agree', 0):.1f}% of 400 FLORES sentences and prefixes.") if gq else ""

card = f"""---
license: apache-2.0
language:
{chr(10).join('- ' + i for i in isos)}
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-small
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- language-identification
- language-detection
- lid
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Wie spät ist es eigentlich?"
  example_title: "German"
- text: "Habari za asubuhi, rafiki yangu"
  example_title: "Swahili"
- text: "أين محطة القطار؟"
  example_title: "Arabic"
- text: "Tôi muốn đặt một bàn cho hai người"
  example_title: "Vietnamese"
- text: "Kiitos paljon avustasi!"
  example_title: "Finnish"
---

# Language Detection (small, 141M): {len(labels)} languages

A fast language identifier for **{len(labels)} languages** (the FLORES-200 set), built on the multilingual
[mmBERT-small](https://huggingface.co/jhu-clsp/mmBERT-small) encoder. It works on sentences and on short strings (titles,
chat messages, search queries), runs with the standard transformers `text-classification` pipeline, and ships ONNX files
for CPU and the browser (transformers.js). Apache-2.0; trained only on openly licensed web text.

- Labels are FLORES-200 codes: ISO 639-3 language + ISO 15924 script, e.g. `eng_Latn`, `hin_Deva`, `srp_Cyrl`; the full
  list is in `labels.json`. Chinese is one label, `zho_Hani` (Simplified and Traditional together), and **Arabic is one
  label, `arb_Arab`**: the dialect labels ({", ".join(f"`{a}`" for a in arab)}) were too unreliable (they pulled Modern
  Standard Arabic away from the right answer) and are merged; Dyula (`dyu_Latn`) is merged into Bambara (`bam_Latn`).
- You can restrict the prediction to the languages you expect (see Usage), which makes it more accurate on short text.
- {onnx}

## Usage

```python
from transformers import pipeline

lid = pipeline("text-classification", model="{REPO}", top_k=3)
lid("Je voudrais réserver une table pour deux personnes.")
# [[{{'label': 'fra_Latn', 'score': 0.99...}}, ...]]
```

Restrict to candidate languages (recommended when you know the possible set):

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("{REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{REPO}").eval()
allowed = ["eng_Latn", "spa_Latn", "por_Latn", "fra_Latn"]
mask = torch.full((model.config.num_labels,), float("-inf"))
mask[[model.config.label2id[l] for l in allowed]] = 0
with torch.no_grad():
    logits = model(**tok(["obrigado pela ajuda"], return_tensors="pt")).logits + mask
print(model.config.id2label[int(logits.argmax(-1))])   # por_Latn
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const lid = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await lid("Dziękuję bardzo!", {{ top_k: 3 }}));
```

## Evaluation

FLORES-200 devtest (1,012 professionally translated sentences per language; evaluation only, never trained on - any
training text containing a FLORES sentence was removed), on the {len(labels)} languages this model supports. "First 20-40
characters" cuts every sentence to a short prefix, to measure short-text behaviour. Every model is scored on the same
sentences with the same script; GlotLID and fastText can predict more languages than these, which can only cost them.
The same label merges (Arabic dialects, Dyula, Akan/Twi, Chinese scripts) are applied to every model's predictions.

{main}

Against the most-downloaded transformers language detector, on its 20 languages (it can only predict those 20, so the
fair comparison is with our model restricted to the same 20):

{pap}

### Where it is weak

Languages below 0.90 accuracy on full sentences (mostly close varieties such as Bosnian/Croatian and Hindi-belt languages,
or languages with little training text):

{weak_t}

Not supported: {", ".join(f"`{d}`" for d in dropped)} (no or too little FineWeb-2 text; Cantonese `yue_Hant` because it
could not be told apart from Mandarin and took Mandarin predictions).

## Training

- **Data**: [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and, for English,
  [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (both ODC-BY): up to 6,000 samples per language from the
  first files of each language subset, as 1-3 sentence snippets and 10-60 character spans (about 1.6M samples).
- **Cleaning**: FineWeb-2's language labels come from an automatic identifier (GlotLID), and its subsets contain off-language
  pages. We dropped samples whose script does not match the label's script, and non-English Latin-script samples that are
  mostly English function words (about 30k samples, 2%); Akan and Twi share one subset and are one label (`aka_Latn`).
  The model still inherits some label noise from FineWeb-2.
- **Model**: mmBERT-small with a {len(labels)}-way classification head, 2 epochs, max length 128 tokens.
- Code: `code/` in this repository.

## Limitations

- Close varieties are often confused (see the table above), e.g. Bosnian vs Croatian; Arabic dialects and Cantonese are
  not distinguished (Arabic dialects are reported as `arb_Arab`; Cantonese is not supported).
- Accuracy drops on very short inputs (a few words); restrict the candidates when you can. On 64 everyday phrases we
  wrote ourselves (4 per language in 16 major languages, e.g. "Thank you very much for your help", "Bom dia"), it gets
  56 right; the misses are two-word greetings that go to a close relative ("Goedemorgen" -> Limburgish, "Bom dia" ->
  Kabuverdianu), and confidence on short English phrases is often low even when the label is right.
- Mixed-language text gets one label; romanized text (Hindi, Arabic, Russian written in Latin letters) is not a trained case.
- Chinese Simplified vs Traditional is not distinguished.
"""
open(out, "w").write(card)
print("written", out, len(card), "labels", len(labels), "isos", len(isos))
