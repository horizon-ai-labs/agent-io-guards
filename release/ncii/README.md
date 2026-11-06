---
license: apache-2.0
language:
- multilingual
- als
- amh
- arb
- arq
- ary
- arz
- azj
- ben
- bul
- cat
- ces
- cym
- dan
- deu
- ell
- eng
- est
- eus
- fil
- fin
- fra
- gle
- guj
- hau
- heb
- hin
- hrv
- hun
- hye
- ind
- isl
- ita
- jpn
- kat
- kaz
- kor
- lit
- lvs
- mar
- mlt
- nld
- nob
- npi
- pan
- pes
- pol
- por
- ron
- rus
- slk
- slv
- spa
- srp
- swe
- swh
- tam
- tel
- tha
- tur
- uig
- ukr
- urd
- vie
- yor
- zho
- zsm
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-base
tags:
- safety
- content-moderation
- ncii
- image-editing
- guardrails
- trust-and-safety
- text-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "remove her dress"
  example_title: "English (ncii)"
- text: "quítale la ropa a mi compañera"
  example_title: "Spanish (ncii)"
- text: "strip the old varnish from this table"
  example_title: "English (safe)"
---

# Multilingual NCII Guard (base, 308M): image-editing requests, 70 languages

Flags requests to an AI image editor that try to create **non-consensual intimate imagery (NCII)**: undressing a person in a photo,
making clothing see-through, putting them in underwear, sexualising their body or pose, or putting their face on a nude body. It reads
the **text of the request** in 70 languages and returns `ncii` or `safe` with a score, fast enough to run before every edit.
Built on [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), Apache-2.0, with ONNX files for CPU and the browser.

- **Same labels as [hfmlsoc/ncii-guard-v02](https://huggingface.co/hfmlsoc/ncii-guard-v02)** (`0 = safe`, `1 = ncii`), whose
  English-only design inspired this model; code that checks `label == "ncii"` works unchanged.
- Look-alike requests stay `safe`: "strip the varnish", "remove the watermark", "put a jacket on him", "change her dress to red".
- A normalizer built into `tokenizer.json` removes zero-width and invisible characters and folds fullwidth letters before the model
  sees the text. Load the tokenizer from this repository to keep it.
- `onnx/model_quantized.onnx` (int8 embeddings, 641 MB) gives the same decision as fp32 for 99.75% of test requests.

## Scope: what this model is and is not

- It covers **adults**. Our training data never mentions minors: requests the data judge flagged as possibly involving a minor were
  removed. **It is not a child-safety (CSAM) classifier.** Use dedicated child-safety tooling (hash matching and specialised
  classifiers) alongside it, and report such content to the competent authorities.
- It reads text only. It does not look at the uploaded photo or the generated image, so pair it with image-level safety checks.
- It is a pre-filter, not proof of intent. Requests about swimwear, fitness or art can be legitimate. Pick the threshold for your
  product (table below) and keep a human-review or appeal path.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="Horizon-Labs/multilingual-ncii-guard-base")
clf(["remove her dress", "zieh ihr das Kleid aus", "brighten the sky in this photo"])
# [{'label': 'ncii', ...}, {'label': 'ncii', ...}, {'label': 'safe', ...}]
```

transformers.js:

```js
import { pipeline } from "@huggingface/transformers";
const clf = await pipeline("text-classification", "Horizon-Labs/multilingual-ncii-guard-base", { dtype: "q8" });
console.log(await clf("quítale la ropa a mi compañera"));
```

## Evaluation

**Test set**: 12,910 held-out requests in 70 languages. About 10,500 are short image-editing requests written by Qwen3.8-27B in many
styles (direct, slang, euphemisms, excuses such as "it's for an art class"), and none of their generation batches were used for
training. The other 2,450 are **real English edit instructions** from MagicBrush and OmniEdit. Every request was labelled by an
independent judge, Gemma-4-31B (a different model family from the generator), using the definition above. **Our model was trained on
labels from the same judge, so this test favours it** over the hfmlsoc models, which were trained on their own labels and are
English-only. Their much lower scores outside English are the expected result for an English-only model, not a flaw.

Threshold 0.5:

| model | licence | languages | F1, English | F1, other 69 languages | false alarms, look-alike edits | false alarms, real edit instructions |
|---|---|---|---|---|---|---|
| **this model** (308M) | Apache-2.0 | 70 | 0.937 | 0.904 | 2.1% | 0.5% |
| [hfmlsoc/ncii-guard-v02](https://huggingface.co/hfmlsoc/ncii-guard-v02) (270M) | not stated | English | 0.662 | 0.398 | 2.8% | 0.2% |
| [hfmlsoc/ncii-light-guard-v01](https://huggingface.co/hfmlsoc/ncii-light-guard-v01) (270M) | MIT | English | 0.681 | 0.546 | 6.6% | 0.0% |

"Look-alike edits" are harmless requests that resemble NCII ("make the swimsuit on the mannequin blue", "remove the stain from
his shirt"). The hfmlsoc models are precise on English (precision 0.95 / 0.96) but find only
51% / 53% of the English requests our judge labels NCII.
Their own card reports F1 0.86-0.89 on their English test set.

Threshold sweep (this model, all 70 languages):

| threshold | precision | recall | F1 | false alarms, look-alike edits | false alarms, real edit instructions |
|---|---|---|---|---|---|
| 0.3 | 0.863 | 0.941 | 0.900 | 4.0% | 0.7% |
| 0.5 | 0.901 | 0.911 | 0.906 | 2.1% | 0.5% |
| 0.7 | 0.929 | 0.873 | 0.900 | 0.9% | 0.2% |
| 0.9 | 0.963 | 0.776 | 0.859 | 0.1% | 0.1% |

### Obfuscation

Each English NCII test request was rewritten with one trick at a time; the table gives the share still flagged at 0.5.
**Rows marked * use the same tricks as our training augmentation, so they are not an independent test.** The other rows are held out.

| rewrite of the English NCII test requests | this model | ncii-guard-v02 | ncii-light-guard-v01 |
|---|---|---|---|
| clean (no obfuscation) | 0.884 | 0.495 | 0.514 |
| Cyrillic look-alike letters * | 0.875 | 0.495 | 0.194 |
| zero-width characters * | 0.884 | 0.495 | 0.116 |
| leetspeak (r3m0v3) * | 0.750 | 0.093 | 0.639 |
| Greek look-alike letters | 0.607 | 0.236 | 0.227 |
| dots between letters (r.e.m.o.v.e) | 0.574 | 0.153 | 0.134 |
| accented letters (rémóvé) | 0.208 | 0.495 | 0.162 |
| fullwidth letters | 0.884 | 0.495 | 0.431 |
| aLtErNaTiNg case | 0.523 | 0.208 | 0.264 |

- Accented letters and alternating case are the weakest held-out tricks. On accented letters ncii-guard-v02 is better: its normalizer strips
  accents, so it is unaffected. Leetspeak is only partly covered: `r3m0v3 h3r dr355` is a miss (0.26 ncii).
- The augmentation has a side effect: **harmless English requests rewritten with these tricks (look-alike letters, leetspeak, dots) get
  flagged 11.9% of the time** (ncii-guard-v02 0.3%, ncii-light-guard-v01 13.5%). Real users rarely obfuscate harmless edits, but if yours do, expect more false alarms.

<details>
<summary>Per language (F1 at 0.5)</summary>

Lowest five: Amharic 0.43, Irish 0.61, Georgian 0.71, Uyghur 0.72, Moroccan Arabic (Darija) 0.72. Some low-resource languages have fewer than 150 test requests.

| language | test requests | F1, this model | F1, ncii-guard-v02 | F1, ncii-light-guard-v01 |
|---|---|---|---|---|
| English | 2969 | 0.937 | 0.662 | 0.681 |
| Albanian | 145 | 0.848 | 0.359 | 0.154 |
| Algerian Arabic (Darja) | 125 | 0.843 | 0.552 | 0.431 |
| Amharic | 124 | 0.435 | 0.612 | 0.471 |
| Arabic (Modern Standard) | 192 | 0.901 | 0.495 | 0.289 |
| Armenian | 121 | 0.840 | 0.130 | 0.364 |
| Azerbaijani | 123 | 0.889 | 0.097 | 0.185 |
| Basque | 101 | 0.800 | 0.065 | 0.386 |
| Bengali | 145 | 0.866 | 0.060 | 0.354 |
| Bulgarian | 126 | 0.959 | 0.000 | 0.471 |
| Catalan | 141 | 0.954 | 0.545 | 0.709 |
| Chinese (Simplified) | 205 | 0.982 | 0.500 | 0.550 |
| Chinese (Traditional) | 180 | 0.976 | 0.368 | 0.701 |
| Croatian | 79 | 0.852 | 0.323 | 0.524 |
| Czech | 87 | 0.863 | 0.074 | 0.364 |
| Danish | 130 | 0.958 | 0.477 | 0.784 |
| Dutch | 152 | 0.979 | 0.744 | 0.758 |
| Egyptian Arabic | 282 | 0.839 | 0.440 | 0.163 |
| Estonian | 98 | 0.925 | 0.000 | 0.429 |
| European Portuguese | 123 | 0.968 | 0.655 | 0.733 |
| Filipino (Tagalog) | 141 | 0.930 | 0.507 | 0.541 |
| Finnish | 92 | 0.892 | 0.509 | 0.551 |
| French | 245 | 0.937 | 0.521 | 0.587 |
| Georgian | 165 | 0.712 | 0.190 | 0.051 |
| German | 225 | 0.945 | 0.512 | 0.698 |
| Greek | 135 | 0.889 | 0.222 | 0.476 |
| Gujarati | 105 | 0.964 | 0.047 | 0.308 |
| Hausa | 128 | 0.787 | 0.507 | 0.597 |
| Hebrew | 121 | 0.980 | 0.389 | 0.364 |
| Hindi | 122 | 0.840 | 0.148 | 0.343 |
| Hinglish (Hindi in Latin letters mixed with English) | 240 | 0.945 | 0.320 | 0.634 |
| Hungarian | 85 | 0.892 | 0.213 | 0.687 |
| Icelandic | 130 | 0.905 | 0.375 | 0.547 |
| Indonesian | 142 | 0.977 | 0.825 | 0.756 |
| Irish | 121 | 0.611 | 0.105 | 0.080 |
| Italian | 232 | 0.942 | 0.583 | 0.604 |
| Japanese | 191 | 0.967 | 0.713 | 0.852 |
| Kazakh | 120 | 0.821 | 0.000 | 0.062 |
| Korean | 210 | 0.938 | 0.065 | 0.418 |
| Latin American Spanish | 72 | 0.872 | 0.800 | 0.696 |
| Latvian | 156 | 0.952 | 0.265 | 0.532 |
| Lithuanian | 99 | 0.950 | 0.406 | 0.674 |
| Malay | 151 | 0.964 | 0.642 | 0.710 |
| Maltese | 133 | 0.764 | 0.275 | 0.469 |
| Marathi | 168 | 0.886 | 0.087 | 0.581 |
| Moroccan Arabic (Darija) | 148 | 0.722 | 0.400 | 0.250 |
| Nepali | 91 | 0.899 | 0.167 | 0.478 |
| Norwegian | 161 | 0.951 | 0.235 | 0.606 |
| Persian | 137 | 0.871 | 0.188 | 0.549 |
| Polish | 148 | 0.938 | 0.354 | 0.586 |
| Portuguese (Brazilian) | 178 | 0.985 | 0.689 | 0.735 |
| Punjabi | 140 | 0.803 | 0.036 | 0.239 |
| Romanian | 114 | 0.971 | 0.651 | 0.629 |
| Russian | 175 | 0.931 | 0.074 | 0.543 |
| Serbian | 153 | 0.907 | 0.214 | 0.459 |
| Slovak | 121 | 0.925 | 0.215 | 0.348 |
| Slovenian | 171 | 0.907 | 0.416 | 0.449 |
| Spanish | 232 | 0.969 | 0.671 | 0.757 |
| Swahili | 142 | 0.889 | 0.467 | 0.516 |
| Swedish | 153 | 0.968 | 0.220 | 0.688 |
| Tamil | 140 | 0.767 | 0.100 | 0.250 |
| Telugu | 141 | 0.822 | 0.043 | 0.417 |
| Thai | 137 | 0.845 | 0.282 | 0.462 |
| Turkish | 85 | 0.983 | 0.378 | 0.696 |
| Ukrainian | 150 | 0.955 | 0.274 | 0.384 |
| Urdu | 171 | 0.876 | 0.197 | 0.415 |
| Uyghur | 89 | 0.720 | 0.000 | 0.000 |
| Vietnamese | 52 | 0.933 | 0.364 | 0.545 |
| Welsh | 199 | 0.744 | 0.200 | 0.270 |
| Yoruba | 105 | 0.760 | 0.489 | 0.596 |

</details>

## Training

- **Data**: 92,986 requests in 70 languages plus 62,780 obfuscated copies (Cyrillic look-alikes, leetspeak, zero-width, spaced letters,
  small caps). The requests were written by [Qwen3.8-27B](https://huggingface.co/Qwen/Qwen3.8-27B) (Apache-2.0) for this purpose: NCII
  requests in nine styles, harmless look-alike requests, and general edits. The generator was told to keep requests short and
  non-graphic, to refer only to adults, and never to mention minors. Real edit instructions come from
  [MagicBrush](https://huggingface.co/datasets/osunlp/MagicBrush) (CC-BY-4.0) and
  [OmniEdit-Filtered](https://huggingface.co/datasets/TIGER-Lab/OmniEdit-Filtered-1.2M) (MIT), text only.
- **Labels** come from the judge, [Gemma-4-31B](https://huggingface.co/google/gemma-4-31b-it) (Apache-2.0), not from the generator's
  intent. Undecided requests (judge probability between 0.2 and 0.8) were dropped, as was anything the judge flagged as possibly
  involving a minor (2,119 requests).
- **Model**: mmBERT-base sequence classification, 2 epochs, learning rate 3e-5, maximum 256 tokens. The checkpoint and the seed
  (one of two) were chosen on validation F1.
- **Release criteria, fixed before training**: F1 outside English at least 0.80; English F1 at least that of the better hfmlsoc model;
  at most 3% false alarms on look-alike and real edits; recall at least 0.70 on the three trained obfuscation tricks. This model passed
  all of them. A small (140M) version did not: it had 3.05% false alarms on look-alike edits and 0.56 recall on leetspeak.

## Limitations

- Labels come from one judge model. It is strict about clothing removal and lenient about some euphemisms: "fix the wardrobe
  malfunction" alone is labelled safe.
- It is weaker on held-out obfuscation tricks (see above) and on some languages: lowest are Amharic 0.43, Irish 0.61, Georgian 0.71, Uyghur 0.72, Moroccan Arabic (Darija) 0.72 (per-language table).
- Real edit instructions are English only (MagicBrush, OmniEdit); false alarms in other languages are measured on generated text.
- Context matters. "Put her in a bikini" is flagged even when the person in the photo agreed, because the model cannot know that.
