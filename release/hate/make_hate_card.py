"""Model card for Horizon-Labs/multilingual-hate-speech-{small,base}.
python release/hate/make_hate_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys
import numpy as np

size, rd, ours_f, other_f, bl_f, seeds, st_f, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-hate-speech-{size}", f"Horizon-Labs/multilingual-hate-speech-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st = first(ours_f), json.load(open(bl_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
T = "Qwen/Qwen3.8-27B (teacher, prompted)"
BL = [("facebook/roberta-hate-speech-dynabench-r4-target", "125M", "none given", "English (Dynabench; the HateCheck authors' lab)"),
      ("cardiffnlp/twitter-roberta-base-hate-latest", "125M", "CC-BY-4.0", "English tweets; trained on 13 hate sets, likely incl. Measuring Hate Speech"),
      ("Hate-speech-CNERG/dehatebert-mono-english", "110M", "Apache-2.0", "English"),
      ("pysentimiento/robertuito-hate-speech::hateful", "109M", "none given", "Spanish tweets"),
      ("IMSyPP/hate_speech_multilingual::!LABEL_0", "178M", "MIT", "multilingual; score = 1 - P(acceptable)"),
      ("christinacdl/XLM_RoBERTa-Multilingual-Hate-Speech-Detection-New", "560M", "MIT", "multilingual; AUC 1.00 on 8 HateCheck languages, so almost surely trained on HateCheck"),
      ("Horizon-Labs/multilingual-toxicity-small::identity_attack", "141M", "Apache-2.0", "our toxicity model, identity_attack label"),
      (T, "27B", "Apache-2.0", "our teacher, for reference")]
link = lambda m: m.split("::")[0] if " " in m else f"[{m.split('::')[0]}](https://huggingface.co/{m.split('::')[0]})"
row = lambda name, p, lic, r, note: (f"| {name} ({p}) | {lic} | {f3(r['_mhc_mean_auc'])} | {f3(r['_mhc_mean_f1'])} | {f3(r.get('_heldout_real_mean_auc'))} | "
                                      f"{f3(r['mhs']['auc'])} | {f3(r['hatexplain']['auc'])} | {note} |")
rows = [row("**this model**", PARAMS[size], "Apache-2.0", me, "multilingual")] + \
       ([row(link(OTHER), PARAMS[other], "Apache-2.0", ot, "")] if ot else []) + [row(link(m), p, lic, bl[m], note) for m, p, lic, note in BL if m in bl]
main = "\n".join(["| model | licence | HateCheck, 11 languages: AUC | HateCheck: F1 @ 0.5 | held-out real comments, 4 sets: mean AUC "
                  "| Measuring Hate Speech (EN)¹: AUC | HateXplain (EN)¹: AUC | note |", "|---|---|---|---|---|---|---|---|"] + rows)
HS = [("hs18", "Stormfront forum (EN)"), ("ethos", "ETHOS: YouTube, Reddit (EN)"), ("olidbr", "OLID-BR (Brazilian Portuguese)"), ("kmhas", "K-MHaS (Korean)")]
held = "\n".join(["| model | " + " | ".join(n for _, n in HS) + " |", "|---" * (len(HS) + 1) + "|"] +
                 [f"| {name} | " + " | ".join(f3(r[k]["auc"]) if k in r else "—" for k, _ in HS) + " |"
                  for name, r in [("**this model**", me)] + [(link(m), bl[m]) for m, *_ in BL if m in bl]])
LN = dict(ara="Arabic", cmn="Mandarin", deu="German", eng="English", fra="French", hin="Hindi", ita="Italian", nld="Dutch", pol="Polish", por="Portuguese", spa="Spanish")
I, F = "IMSyPP/hate_speech_multilingual::!LABEL_0", "facebook/roberta-hate-speech-dynabench-r4-target"
langs = sorted(k[4:] for k in me if k.startswith("mhc_"))
per = "\n".join(["| language | this model: AUC | F1 @ 0.5 | IMSyPP multilingual: AUC | facebook r4 (English): AUC | teacher: AUC |", "|---|---|---|---|---|---|"] +
                [f"| {LN.get(l, l)} | {f3(me['mhc_' + l]['auc'])} | {f3(me['mhc_' + l]['f1_05'])} | {f3(bl[I]['mhc_' + l]['auc'])} | {f3(bl[F]['mhc_' + l]['auc'])} | "
                 f"{f3(bl[T]['mhc_' + l]['auc'])} |" for l in langs])
func = lambda r: {fn: float(np.mean([r[f"mhc_{l}"]["func_acc"][fn] for l in langs if fn in r[f"mhc_{l}"]["func_acc"]]))
                  for fn in sorted({fn for l in langs for fn in r[f"mhc_{l}"]["func_acc"]})}
fm, ft = func(me), func(bl[T])
FN = {"counter_quote_nh": "counter-speech quoting hate", "counter_ref_nh": "counter-speech referencing hate", "negate_neg_nh": "negated hate",
      "ident_neutral_nh": "neutral mention of a group", "ident_pos_nh": "positive statement about a group", "profanity_nh": "profanity, not hateful",
      "slur_reclaimed_nh": "reclaimed slurs", "slur_homonym_nh": "slur homonyms", "target_obj_nh": "abuse aimed at objects",
      "target_indiv_nh": "abuse aimed at individuals", "target_group_nh": "abuse aimed at non-protected groups"}
nh = "\n".join(["| non-hateful test type | this model: accuracy @ 0.5 | teacher |", "|---|---|---|"] +
               [f"| {FN.get(k, k)} | {fm[k]:.2f} | {ft.get(k, float('nan')):.2f} |" for k in fm if k.endswith("_nh")])
hate_acc = np.mean([v for k, v in fm.items() if k.endswith("_h")])
C = "christinacdl/XLM_RoBERTa-Multilingual-Hate-Speech-Detection-New"
fair = [m for m, *_ in BL if m in bl and m not in (C, T)]
lead = [LN[l] for l in langs if me["mhc_" + l]["auc"] > max(bl[m]["mhc_" + l]["auc"] for m in fair)]
lead_txt = ("leads in every HateCheck language" if len(lead) == len(langs) else f"leads in {len(lead)} of {len(langs)} HateCheck languages ({', '.join(lead)})")
real_ahead = [m.split("::")[0].split("/")[1] for m, *_ in BL if m in bl and m != T and bl[m].get("_heldout_real_mean_auc", 0) > me["_heldout_real_mean_auc"]]
tg = me["mhs"].get("target_auc", {})
tgt = ", ".join(f"{g} {v:.2f}" for g, v in tg.items() if v is not None)
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB): probabilities differ from fp32 by {gq.get('mean_diff', 0):.4f} on average "
        f"over {qs.get('n', 0)} test texts.") if gq else ""

card = f"""---
license: apache-2.0
language:
- multilingual
- en
- es
- fr
- de
- it
- pt
- nl
- pl
- cs
- ro
- hu
- el
- bg
- sr
- hr
- uk
- ru
- sv
- da
- 'no'
- fi
- tr
- ar
- he
- fa
- ur
- hi
- bn
- ta
- te
- mr
- zh
- ja
- ko
- vi
- th
- id
- ms
- tl
- sw
- am
- ha
- yo
- af
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
tags:
- hate-speech
- hate-speech-detection
- content-moderation
- trust-and-safety
- multi-label-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Immigrants are vermin and should be thrown out of the country."
  example_title: "Hateful"
- text: "Saying that immigrants are vermin is disgusting and wrong."
  example_title: "Counter-speech"
- text: "You are a complete idiot."
  example_title: "Insult, not hate"
- text: "Die sind alle Ungeziefer, schmeißt sie raus."
  example_title: "German"
---

# Multilingual Hate Speech Detector ({size}, {PARAMS[size]})

Detects **hate speech**, meaning attacks on people because of a protected characteristic, in English and about 50 other languages.
It also says **which group is targeted**. Outputs (independent sigmoid probabilities):

- **`hate`**: the text attacks, demeans, dehumanises, stereotypes or calls for harm or exclusion of people because of their race or
  ethnicity, nationality or migration status, religion, gender, sexual orientation, gender identity, disability or age;
- **`target_race`, `target_religion`, `target_origin`, `target_gender`, `target_sexuality`, `target_age`, `target_disability`**:
  which group is attacked (meaningful when `hate` is high).

Ordinary insults, profanity, attacks on politicians, companies or sports fans, criticism of ideas, and counter-speech are *not*
hate speech here, so it complements a general toxicity model such as our
[multilingual-toxicity](https://huggingface.co/Horizon-Labs/multilingual-toxicity-small).
Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, ONNX for CPU and the browser (transformers.js) included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- Flag with `hate >= 0.5`, or tune the threshold on your own data. {onnx}

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)
print(clf("Die sind alle Ungeziefer, schmeißt sie raus."))
# [[{{'label': 'hate', 'score': ...}}, {{'label': 'target_origin', 'score': ...}}, ...]]
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await clf("Immigrants are vermin.", {{ top_k: null }}));
```

## Evaluation

Every model is scored the same way (`code/hate/eval_hate.py`): ROC AUC of its hate probability, and F1 at 0.5. No test item was
used for training.

- **Multilingual HateCheck** ([mteb/multi-hatecheck](https://huggingface.co/datasets/mteb/multi-hatecheck), CC-BY-4.0): 40,310 functional
  test cases in 11 languages, including hard non-hateful cases (counter-speech, negation, abuse of individuals or objects, neutral
  mentions of groups).
- **Held-out real comments** (no text from these sets was used in training): Stormfront forum sentences
  ([hate_speech18](https://huggingface.co/datasets/odegiber/hate_speech18), CC-BY-SA-3.0), [ETHOS](https://github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset)
  binary (YouTube and Reddit), [OLID-BR](https://huggingface.co/datasets/dougtrajano/olid-br) test (Brazilian Portuguese; racism,
  LGBTQphobia, religious intolerance, sexism or xenophobia count as hate) and [K-MHaS](https://github.com/adlnlp/K-MHaS) (Korean news
  comments, 5,000; attacks on protected groups vs not hate speech).
- ¹ **In-domain**: [Measuring Hate Speech](https://huggingface.co/datasets/ucberkeley-dlab/measuring-hate-speech) (6,000 comments,
  hateful = the authors' score > 0.5) and the [HateXplain](https://github.com/hate-alert/HateXplain) test split (offensive-but-not-hate
  counts as negative). Other texts from these two datasets (not the test items, and not their labels) are in our training data.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- The English baselines do well on English but are near chance on the other HateCheck languages. Among models not trained on
  HateCheck, this model {lead_txt}. On the held-out real-comment mean, models ahead of it: {", ".join(real_ahead) or "none"} (teacher excluded).
  The XLM-R-large model's near-perfect HateCheck score comes with no training-data documentation; its real-comment scores are the
  comparable evidence.
- Caveat: our synthetic training data deliberately includes counter-speech, quoted hate, offensive-but-not-hateful text and implicit
  hate, which HateCheck also tests (no HateCheck items or templates were used). The held-out real comments are the more independent test.

Held-out real comments by set (AUC):

{held}

Per HateCheck language:

{per}

HateCheck by test type (mean over languages): hateful cases detected {hate_acc:.2f}; non-hateful cases correctly left alone:

{nh}

Target groups (Measuring Hate Speech, hateful comments, AUC of each `target_*` output vs the annotated target): {tgt}.

## Training

- **Texts**: {st['by_source']['civil']:,} comments from [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (CC0) and
  their machine translations from [Horizon-Labs/civil-comments-multilingual](https://huggingface.co/datasets/Horizon-Labs/civil-comments-multilingual)
  (half selected for identity-related attacks), plus {st['by_source']['gen']:,} texts written natively in 52 languages by Qwen3.8-27B
  (Apache-2.0) for this purpose: hateful texts against groups of 7 protected categories (explicit and implicit or coded), non-hateful
  texts mentioning those groups (neutral, news-style, counter-speech, quoted hate), and offensive texts that are not hate speech.
  Plus {st['by_source']['mhs'] + st['by_source']['hatexplain']:,} real social-media posts: Measuring Hate Speech comments outside our 6,000-comment
  test sample, and the HateXplain train and validation splits (CC-BY-4.0), counted twice. Only their text was used; the labels came
  from the teacher.
- **Labels**: Qwen3.8-27B (prompted with the definition above) answered one yes/no question per output for every text; the model learns
  these probabilities (soft labels, binary cross-entropy). The checkpoint was chosen on {st['val']:,} held-out teacher-labelled texts.
- mmBERT-{size}, max length 256 tokens. Code: `code/` in this repository.

## Limitations

- What counts as hate speech depends on law, platform policy and culture. This model follows one definition (above). Check it
  against your own policy and data before relying on it.
- Labels come from an LLM teacher, not human annotators, and its errors carry over. The teacher's own scores are in the table.
- Weaker on implicit hate, on Arabic and Hindi (see the per-language table), and on languages and dialects that are rare in the data.
- Do not use it as the only basis for decisions about people. Keep a human in the loop for enforcement.
"""
open(out, "w").write(card)
print("written", out, len(card))
