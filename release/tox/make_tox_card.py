"""Model card for Horizon-Labs/multilingual-toxicity-{small,base}.
python release/tox/make_tox_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, seeds, st_f, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-toxicity-{size}", f"Horizon-Labs/multilingual-toxicity-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st = first(ours_f), json.load(open(bl_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
BL = [("unitary/unbiased-toxic-roberta", "125M", "Apache-2.0", "7 Detoxify labels (+ identity labels), English"),
      ("unitary/toxic-bert", "110M", "Apache-2.0", "6 labels, English"), ("s-nlp/roberta_toxicity_classifier", "125M", "OpenRAIL++", "binary, English"),
      ("martin-ha/toxic-comment-model", "67M", "none given", "binary, English"),
      ("unitary/multilingual-toxic-xlm-roberta", "278M", "Apache-2.0", "1 label, multilingual"),
      ("citizenlab/distilbert-base-multilingual-cased-toxicity", "135M", "none given", "binary, multilingual"),
      ("textdetox/xlmr-large-toxicity-classifier", "560M", "OpenRAIL++", "binary, multilingual; trained on TextDetox data")]
link = lambda m: f"[{m}](https://huggingface.co/{m})"
row = lambda name, lic, r, note: f"| {name} | {lic} | {f3(r['_civil_mean_auc'])} | {f3(r['_civil_toxicity_auc'])} | {f3(r['_tdx_mean_auc'])} | {f3(r['_tdx_mean_f1'])} | {note} |"
rows = [row(f"**this model** ({PARAMS[size]})", "Apache-2.0", me, "7 Detoxify labels, multilingual")] + \
       ([row(f"{link(OTHER)} ({PARAMS[other]})", "Apache-2.0", ot, "")] if ot else []) + [row(f"{link(m)} ({p})", lic, bl[m], note) for m, p, lic, note in BL]
main = "\n".join(["| model | licence | Civil Comments, mean AUC (6 labels) | Civil Comments, toxicity AUC | TextDetox (15 languages), AUC | TextDetox, F1 @ 0.5 | labels |",
                  "|---|---|---|---|---|---|---|"] + rows)
LN = dict(am="Amharic", ar="Arabic", de="German", en="English", es="Spanish", fr="French", he="Hebrew", hi="Hindi", hin="Hinglish (romanised Hindi)",
          it="Italian", ja="Japanese", ru="Russian", tt="Tatar", uk="Ukrainian", zh="Chinese")
X, U = "textdetox/xlmr-large-toxicity-classifier", "unitary/multilingual-toxic-xlm-roberta"
langs = sorted(k[4:] for k in me if k.startswith("tdx_"))
per = "\n".join(["| language | this model (AUC) | multilingual-toxic-xlm-roberta | xlmr-large-toxicity (in-domain) |", "|---|---|---|---|"] +
                [f"| {LN.get(l, l)} | {f3(me['tdx_' + l]['auc'])} | {f3(bl[U]['tdx_' + l]['auc'])} | {f3(bl[X]['tdx_' + l]['auc'])} |" for l in langs])
lab = "\n".join(["| label | this model | unbiased-toxic-roberta |", "|---|---|---|"] +
                [f"| {l} | {f3(me['civil_test']['auc'].get(l))} | {f3(bl['unitary/unbiased-toxic-roberta']['civil_test']['auc'].get(l))} |" for l in me["civil_test"]["auc"]])
ahead = {k: [m.split("/")[1] for m, *_ in BL if bl[m][k] > me[k]] for k in ["_civil_mean_auc", "_civil_toxicity_auc", "_tdx_mean_auc"]}
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB): probabilities differ from fp32 by {gq.get('mean_diff', 0):.4f} on average "
        f"over {qs.get('n', 0)} test texts.") if gq else ""

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
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- google/civil_comments
tags:
- toxicity
- toxic-comment-classification
- content-moderation
- detoxify
- multi-label-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Thanks for sharing, this was really helpful!"
  example_title: "Harmless"
- text: "You are a complete idiot and nobody wants you here."
  example_title: "Insult"
- text: "Halt die Klappe, du Vollidiot."
  example_title: "German"
- text: "Eres un inútil, lárgate de aquí."
  example_title: "Spanish"
---

# Multilingual Toxicity ({size}, {PARAMS[size]}): Detoxify labels in many languages

A multi-label toxic-comment classifier with the labels of [Detoxify](https://github.com/unitaryai/detoxify) /
[unitary/unbiased-toxic-roberta](https://huggingface.co/unitary/unbiased-toxic-roberta): **toxicity, severe_toxicity,
obscene, threat, insult, identity_attack, sexual_explicit**. It works on comments in English and 33 other languages.
Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}) and trained on Civil Comments (CC0) plus translations,
so it is a multilingual drop-in for Detoxify-style moderation. Apache-2.0. ONNX files for CPU and the browser
(transformers.js) are included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- Each label gets an independent probability (sigmoid), as in Detoxify. A common choice is to flag a comment when
  `toxicity >= 0.5`, but tune the threshold for your platform. Scores for non-English text tend to be lower than for
  English (see F1 @ 0.5 below), so a lower threshold may suit multilingual content. Check it on your own data.
- For harmful requests to an LLM (weapons, self-harm, etc.) rather than rude comments, see our
  [content-safety-guard](https://huggingface.co/Horizon-Labs/content-safety-guard-small).
- {onnx}

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)
print(clf("Halt die Klappe, du Vollidiot."))
# [[{{'label': 'toxicity', 'score': ...}}, {{'label': 'insult', 'score': ...}}, ...]]
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await clf("Eres un inútil, lárgate de aquí.", {{ top_k: null }}));
```

## Evaluation

The benchmarks were used only for evaluation, and training texts that also occur in them were removed.

- **Civil Comments test** (English, 20,000 random comments): ROC AUC per label, with labels binarised at >= 0.5 as in the
  Jigsaw competitions. The table gives the mean over the 6 labels that have at least 20 positives (severe_toxicity
  has none at that threshold), and the toxicity AUC. Models without a label are scored on the labels they have.
- **TextDetox** ([textdetox/multilingual_toxicity_dataset](https://huggingface.co/datasets/textdetox/multilingual_toxicity_dataset)):
  binary toxic/non-toxic in 15 languages, 1,000 balanced texts each. The score is ROC AUC of each model's toxicity
  probability, plus F1 at 0.5.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- Models ahead of this one: Civil Comments mean AUC: {", ".join(ahead["_civil_mean_auc"]) or "none"} (single-label models are
  scored on their one label, so compare the toxicity column too); Civil Comments toxicity AUC: {", ".join(ahead["_civil_toxicity_auc"]) or "none"}; TextDetox AUC:
  {", ".join(ahead["_tdx_mean_auc"]) or "none"}. The English models were trained on Jigsaw/Civil Comments data, the same
  source as this test set. xlmr-large-toxicity-classifier was trained on the TextDetox data.

Per label (Civil Comments, AUC):

{lab}

Per TextDetox language (AUC):

{per}

## Training

- **Data**: {st['n_en']:,} English comments from [Civil Comments](https://huggingface.co/datasets/google/civil_comments)
  (CC0; enriched for toxic ones), with their fractional annotator labels. Qwen3.8-27B (Apache-2.0) translated
  {st['n_translated']:,} of them into 33 languages, about 12,000 per language and half of them toxic. It was told to
  keep insults, profanity and threats intact, and the labels are copied to each translation.
- **Model**: mmBERT-{size} with 7 sigmoid outputs, binary cross-entropy on the fractional labels (as in Detoxify), max
  length 256 tokens. The checkpoint was chosen by mean AUC on held-out comments and their translations.
- Code: `code/` in this repository.

## Limitations

- Toxicity labels are subjective and culture-dependent. Translated comments keep English annotators' judgements, and
  slurs or insults that exist only in other languages are under-represented.
- Like other toxicity models, it can over-flag mentions of identity groups and reclaimed or quoted language, and miss
  implicit or sarcastic abuse. Do not use it as the only basis for decisions about people.
- Accuracy is lower for low-resource languages, where both translation and the base model are weaker (see the per-language table).
"""
open(out, "w").write(card)
print("written", out, len(card))
