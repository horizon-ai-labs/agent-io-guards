"""Model card for Horizon-Labs/multilingual-emotions-{small,base}.
python release/emotion/make_emo_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json BASELINES.json STATS.json SEEDS.json OUT"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, st_f, seeds_f, out = sys.argv[1:9]
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "emotion"))
from common import GO, EKMAN
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-emotions-{size}", f"Horizon-Labs/multilingual-emotions-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, ot, bl, st, seeds = first(ours_f), first(other_f), json.load(open(bl_f)), json.load(open(st_f)), json.load(open(seeds_f))
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
BL = [("SamLowe/roberta-base-go_emotions", "125M", "MIT", "28 GoEmotions labels, English"),
      ("AnasAlokla/multilingual_go_emotions", "", "MIT", "28 GoEmotions labels, multilingual"),
      ("j-hartmann/emotion-english-distilroberta-base", "82M", "none given", "7 labels, English"),
      ("MilaNLProc/xlm-emo-t", "278M", "none given", "4 labels (anger, fear, joy, sadness), multilingual tweets"),
      ("tabularisai/multilingual-emotion-classification", "135M", "CC-BY-NC-4.0", "11 labels, multilingual")]
g = lambda r, k: r.get("goemo", {}).get(k)
row = lambda name, lic, r, note: (f"| {name} | {lic} | {f3(g(r, 'macro_f1_05'))} | {f3(g(r, 'macro_f1_tuned'))} | "
                                  f"{f3(r['_brighter_mean_f1']) if len(r['covered']) == 6 else '—'} | {f3(r['_brighter_mean_f1_4'])} | {note} |")
rows = [row(f"**this model** ({PARAMS[size]})", "Apache-2.0", me, "28 GoEmotions labels, multilingual").replace(f"| {f3(g(me, 'macro_f1_tuned'))} |", f"| **{f3(g(me, 'macro_f1_tuned'))}** |", 1),
        row(f"[{OTHER.split('/')[1]}](https://huggingface.co/{OTHER}) ({PARAMS[other]})", "Apache-2.0", ot, "")]
for m, p, lic, note in BL:
    rows.append(row(f"[{m}](https://huggingface.co/{m})" + (f" ({p})" if p else ""), lic, bl[m], note))
main = "\n".join(["| model | licence | GoEmotions test, macro-F1 @0.5 | GoEmotions test, macro-F1 (tuned thresholds) | BRIGHTER, 6 emotions | BRIGHTER, 4 emotions | labels |",
                  "|---|---|---|---|---|---|---|"] + rows)
LN = dict(afr="Afrikaans", arq="Algerian Arabic", ary="Moroccan Arabic", chn="Chinese", deu="German", eng="English", esp="Spanish",
          hau="Hausa", hin="Hindi", ibo="Igbo", ind="Indonesian", jav="Javanese", kin="Kinyarwanda", mar="Marathi", pcm="Nigerian Pidgin",
          ptbr="Portuguese (Brazil)", ptmz="Portuguese (Mozambique)", ron="Romanian", rus="Russian", sun="Sundanese", swa="Swahili",
          swe="Swedish", tat="Tatar", ukr="Ukrainian", vmw="Makhuwa", xho="isiXhosa", yor="Yoruba", zul="isiZulu")
X, T = "MilaNLProc/xlm-emo-t", "tabularisai/multilingual-emotion-classification"
langs = sorted(k[9:] for k in me if k.startswith("brighter_") and me[k].get("macro_f1") is not None)
per = "\n".join(["| language | this model (6 emotions) | this model (4) | xlm-emo-t (4) | tabularisai (6, NC) |", "|---|---|---|---|---|"] +
                [f"| {LN.get(l, l)} | {f3(me['brighter_' + l]['macro_f1'])} | {f3(me['brighter_' + l]['macro_f1_4'])} | "
                 f"{f3(bl[X]['brighter_' + l]['macro_f1_4'])} | {f3(bl[T]['brighter_' + l]['macro_f1'])} |" for l in langs])
trained = set(k for k in st["per_lang"]) | {"English"}
ahead = [m.split("/")[1] for m, *_ in BL if bl[m]["_brighter_mean_f1_4"] > me["_brighter_mean_f1_4"]]
ahead6 = [m.split("/")[1] for m, *_ in BL if len(bl[m]["covered"]) == 6 and bl[m]["_brighter_mean_f1"] > me["_brighter_mean_f1"]]
aheadg = [m.split("/")[1] for m, *_ in BL if g(bl[m], "macro_f1_tuned") and g(bl[m], "macro_f1_tuned") > g(me, "macro_f1_tuned")]
EP = {"small": "1.5 epochs", "base": "1 epoch"}[size]
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB) agrees with fp32 on {100 * gq.get('agree', 0):.1f}% of "
        f"{qs.get('n', 0)} test texts (all 28 labels thresholded at 0.5).") if gq else ""
thr = me["goemo"]["thresholds"]
worst = sorted(me["goemo"]["per_label_tuned"].items(), key=lambda x: x[1])[:6]

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
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- google-research-datasets/go_emotions
tags:
- emotion
- emotion-classification
- go_emotions
- multi-label-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Thank you so much, this made my day!"
  example_title: "English"
- text: "Ich kann nicht glauben, dass sie das Konzert abgesagt haben. So eine Frechheit."
  example_title: "German"
- text: "Tengo miedo de que no lleguemos a tiempo."
  example_title: "Spanish"
- text: "我真的很想念我的奶奶。"
  example_title: "Chinese"
- text: "Wow, I did not expect that at all!"
  example_title: "Surprise"
---

# Multilingual Emotions ({size}, {PARAMS[size]}): the 28 GoEmotions labels in many languages

A multi-label emotion classifier with the 28 labels of Google's [GoEmotions](https://huggingface.co/datasets/google-research-datasets/go_emotions)
(admiration, amusement, anger, ... , surprise, neutral). It works on text in English and 35 other languages. It is a drop-in
multilingual alternative to English-only GoEmotions models such as
[SamLowe/roberta-base-go_emotions](https://huggingface.co/SamLowe/roberta-base-go_emotions): the labels are the same and it
uses the same sigmoid multi-label output. Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}),
Apache-2.0. ONNX files for CPU and the browser (transformers.js) are included.
{"A larger, more accurate version is available as" if size == "small" else "A smaller, faster version is available as"}
[{OTHER.split('/')[1]}](https://huggingface.co/{OTHER}). [Try it in the browser](https://huggingface.co/spaces/Horizon-Labs/multilingual-emotions).

- Each label gets an independent probability. A text can have several emotions, or only `neutral`.
- `thresholds.json` has one threshold per label, tuned on the GoEmotions validation set. Use it instead of 0.5,
  especially for rare labels (grief, pride, relief, nervousness). See Usage.
- For the 6 Ekman emotions (anger, disgust, fear, joy, sadness, surprise), group the labels with the GoEmotions paper's
  mapping and take the maximum per group (`ekman_mapping.json`, code below).
- {onnx}

## Usage

```python
import json
from huggingface_hub import hf_hub_download
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)
thr = json.load(open(hf_hub_download("{REPO}", "thresholds.json")))
out = clf(["Thank you so much, this made my day!", "Tengo miedo de que no lleguemos a tiempo."])
print([[x["label"] for x in o if x["score"] >= thr[x["label"]]] for o in out])
# e.g. [['excitement', 'gratitude', 'joy'], ['fear']]

# 6 Ekman emotions: max score over each group
ekman = json.load(open(hf_hub_download("{REPO}", "ekman_mapping.json")))
scores = {{x["label"]: x["score"] for x in out[1]}}
print({{e: max(scores[l] for l in ls) for e, ls in ekman.items()}})
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await clf("Je suis tellement fier de toi !", {{ top_k: 5 }}));
```

## Evaluation

The benchmarks were used only for evaluation, and training texts that also occur in them were removed.

- **GoEmotions test** (English Reddit comments, 5,427, 28 labels). Macro-F1 over the 28 labels, at threshold 0.5 and with
  per-label thresholds tuned on the GoEmotions validation set. Only models with the 28 GoEmotions labels are scored.
- **BRIGHTER** ([brighter-dataset/BRIGHTER-emotion-categories](https://huggingface.co/datasets/brighter-dataset/BRIGHTER-emotion-categories),
  CC-BY-4.0): human-labelled texts written in 28 languages (not translations), with 6 emotions and multiple labels
  per text; up to 1,500 test texts per language. Every model's labels are grouped into the 6 emotions (GoEmotions labels via the
  GoEmotions paper's Ekman mapping; for other label sets, by name). Each model gets one threshold per language and
  emotion, tuned on that language's BRIGHTER dev set. The score is macro-F1 over the annotated emotions, averaged over
  languages. "4 emotions" = anger, fear, joy and sadness only, the set every compared model covers.

{main}

- The table shows the released checkpoint. Means over {seeds["x"]}.
- Models ahead of this one: GoEmotions (tuned): {", ".join(aheadg) or "none"}; BRIGHTER 6 emotions: {", ".join(ahead6) or "none"};
  BRIGHTER 4 emotions: {", ".join(ahead) or "none"}. Most BRIGHTER languages are
  low-resource African and Asian languages that are not in our training data; GoEmotions texts are Reddit comments,
  while BRIGHTER includes tweets, news comments and other sources.

Per BRIGHTER language (macro-F1):

{per}

Weakest GoEmotions labels (test F1, tuned thresholds): {", ".join(f"{l} {v:.2f}" for l, v in worst)}. Several of these are rare,
and GoEmotions annotators often disagree on such labels.

## Training

- **Data**: the GoEmotions training set (43,410 English Reddit comments with human labels; Apache-2.0), plus translations
  of it into 35 languages made with Qwen3.8-27B (Apache-2.0). Each language gets its own random sample of 12,000 comments,
  and the human labels are copied to the translation ({st['n_train']:,} training examples in total). The languages are
  {", ".join(sorted(trained - {"English"}))}.
- Translations that failed to parse, looked like refusals, had an implausible length or were copied unchanged were
  dropped (about 1.2%). 3% of the source comments (in every language) were held out for validation.
- **Model**: mmBERT-{size} with 28 sigmoid outputs and binary cross-entropy, {EP}, learning rate 5e-5, max length 256
  tokens. Epochs and learning rate were chosen on the GoEmotions validation set and BRIGHTER dev sets, never the test sets.
  The checkpoint was chosen by macro-F1 on held-out training comments.
- Code: `code/` in this repository.

## Limitations

- The labels come from English Reddit comments. Translations keep the labels but can shift nuance, and the model has
  seen little native (non-translated) emotional text outside English. BRIGHTER results show it is much weaker in
  languages it was not trained on.
- The labels are subjective and inter-annotator agreement on GoEmotions is modest, so F1 around 0.5 is typical for this
  task. Rare labels are unreliable.
- It classifies the emotion expressed in the text, not the writer's actual state. Don't use it to make decisions
  about individuals.
"""
open(out, "w").write(card)
json.dump({l: round(v, 3) for l, v in thr.items()}, open(os.path.join(os.path.dirname(out), f"thresholds_{size}.json"), "w"), indent=1)
print("written", out, len(card))
