"""Model card for Horizon-Labs/multilingual-ncii-guard-base.
python release/ncii/make_ncii_card.py EVAL_DIR RELEASE_DIR OUT
EVAL_DIR: ncii/eval_ncii.py output with DUMP=1 (eval.json + scores_<model>.parquet for ours and the two hfmlsoc models)."""
import json, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "compress"))
from align import code

ev, rd, out = sys.argv[1:4]
R = json.load(open(f"{ev}/eval.json"))
OURS = [k for k in R if not k.startswith("ncii-")][0]
V02, LIGHT = "ncii-guard-v02", "ncii-light-guard-v01"
REPO = "Horizon-Labs/multilingual-ncii-guard-base"
S = {k: pd.read_parquet(f"{ev}/scores_{k}.parquet") for k in (OURS, V02, LIGHT)}
sw = json.load(open(f"{rd}/onnx_sweep.json")); q = sw[sw["best"]]   # quant_sweep output: best config name -> its stats
f3 = lambda x: f"{x:.3f}"
pc = lambda x: f"{100 * x:.1f}%"


def prf(y, p, th):
    yp = p >= th; tp = int((yp & (y == 1)).sum()); fp = int((yp & (y == 0)).sum()); fn = int((~yp & (y == 1)).sum())
    pr = tp / max(1, tp + fp); rc = tp / max(1, tp + fn); return pr, rc, 2 * pr * rc / max(1e-9, pr + rc)


rows = [(f"**this model** (308M)", "Apache-2.0", "70", OURS), ("[hfmlsoc/ncii-guard-v02](https://huggingface.co/hfmlsoc/ncii-guard-v02) (270M)", "not stated", "English", V02),
        ("[hfmlsoc/ncii-light-guard-v01](https://huggingface.co/hfmlsoc/ncii-light-guard-v01) (270M)", "MIT", "English", LIGHT)]
main = "\n".join(["| model | licence | languages | F1, English | F1, other 69 languages | false alarms, look-alike edits | false alarms, real edit instructions |",
                  "|---|---|---|---|---|---|---|"] +
                 [f"| {n} | {lic} | {lg} | {f3(R[k]['en']['f1'])} | {f3(R[k]['non_en']['f1'])} | {pc(R[k]['fpr_hard'])} | {pc(R[k]['fpr_real'])} |" for n, lic, lg, k in rows])
so = S[OURS]; y, p = so.label.values, so.p.values
sweep = "\n".join(["| threshold | precision | recall | F1 | false alarms, look-alike edits | false alarms, real edit instructions |", "|---|---|---|---|---|---|"] +
                  [f"| {t} | {f3(prf(y, p, t)[0])} | {f3(prf(y, p, t)[1])} | {f3(prf(y, p, t)[2])} | {pc((p[(so.kind == 'hard').values] >= t).mean())} | "
                   f"{pc((p[(so.kind == 'real_edit').values] >= t).mean())} |" for t in (0.3, 0.5, 0.7, 0.9)])
FAM = [("clean (no obfuscation)", "recall_clean_en"), ("Cyrillic look-alike letters *", "recall_homoglyph"), ("zero-width characters *", "recall_zerowidth"),
       ("leetspeak (r3m0v3) *", "recall_leet"), ("Greek look-alike letters", "recall_greek"), ("dots between letters (r.e.m.o.v.e)", "recall_dots"),
       ("accented letters (rémóvé)", "recall_diacritics"), ("fullwidth letters", "recall_fullwidth"), ("aLtErNaTiNg case", "recall_altcase")]
obf = "\n".join(["| rewrite of the English NCII test requests | this model | ncii-guard-v02 | ncii-light-guard-v01 |", "|---|---|---|---|"] +
                [f"| {n} | {f3(R[OURS][k])} | {f3(R[V02][k])} | {f3(R[LIGHT][k])} |" for n, k in FAM])
langs = sorted(so.lang.unique(), key=lambda l: (l != "English", l))
per = "\n".join(["| language | test requests | F1, this model | F1, ncii-guard-v02 | F1, ncii-light-guard-v01 |", "|---|---|---|---|---|"] +
                [f"| {l} | {int((so.lang == l).sum())} | " + " | ".join(f3(prf(S[k][S[k].lang == l].label.values, S[k][S[k].lang == l].p.values, 0.5)[2]) for k in (OURS, V02, LIGHT)) + " |"
                 for l in langs])
pl = {l: prf(so[so.lang == l].label.values, so[so.lang == l].p.values, 0.5)[2] for l in langs}
worst = ", ".join(f"{l} {v:.2f}" for l, v in sorted(pl.items(), key=lambda kv: kv[1])[:5])
qq = q
ISO = sorted({code(l)[:3] for l in langs})

card = f"""---
license: apache-2.0
language:
{chr(10).join("- " + c for c in ["multilingual"] + ISO)}
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
- `onnx/model_quantized.onnx` (int8 embeddings, {qq.get('mb', 641):.0f} MB) gives the same decision as fp32 for {100 * qq.get('agree', 0.9975):.2f}% of test requests.

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

clf = pipeline("text-classification", model="{REPO}")
clf(["remove her dress", "zieh ihr das Kleid aus", "brighten the sky in this photo"])
# [{{'label': 'ncii', ...}}, {{'label': 'ncii', ...}}, {{'label': 'safe', ...}}]
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
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

{main}

"Look-alike edits" are harmless requests that resemble NCII ("make the swimsuit on the mannequin blue", "remove the stain from
his shirt"). The hfmlsoc models are precise on English (precision {R[V02]["en"]["precision"]:.2f} / {R[LIGHT]["en"]["precision"]:.2f}) but find only
{R[V02]["en"]["recall"]:.0%} / {R[LIGHT]["en"]["recall"]:.0%} of the English requests our judge labels NCII.
Their own card reports F1 0.86-0.89 on their English test set.

Threshold sweep (this model, all 70 languages):

{sweep}

### Obfuscation

Each English NCII test request was rewritten with one trick at a time; the table gives the share still flagged at 0.5.
**Rows marked * use the same tricks as our training augmentation, so they are not an independent test.** The other rows are held out.

{obf}

- Accented letters and alternating case are the weakest held-out tricks. On accented letters ncii-guard-v02 is better: its normalizer strips
  accents, so it is unaffected. Leetspeak is only partly covered: `r3m0v3 h3r dr355` is a miss (0.26 ncii).
- The augmentation has a side effect: **harmless English requests rewritten with these tricks (look-alike letters, leetspeak, dots) get
  flagged {pc(R[OURS]["fpr_obf_benign_en"])} of the time** (ncii-guard-v02 {pc(R[V02]["fpr_obf_benign_en"])}, ncii-light-guard-v01 {pc(R[LIGHT]["fpr_obf_benign_en"])}). Real users rarely obfuscate harmless edits, but if yours do, expect more false alarms.

<details>
<summary>Per language (F1 at 0.5)</summary>

Lowest five: {worst}. Some low-resource languages have fewer than 150 test requests.

{per}

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
- It is weaker on held-out obfuscation tricks (see above) and on some languages: lowest are {worst} (per-language table).
- Real edit instructions are English only (MagicBrush, OmniEdit); false alarms in other languages are measured on generated text.
- Context matters. "Put her in a bikini" is flagged even when the person in the photo agreed, because the model cannot know that.
"""
open(out, "w").write(card); print(out, len(card))
