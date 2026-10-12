"""Model card for Horizon-Labs/multilingual-edu-classifier-{small,base}.
python release/edu/make_edu_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json TEACHER.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys
import numpy as np

size, rd, ours_f, other_f, bl_f, te_f, seeds, st_f, out = sys.argv[1:10]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-edu-classifier-{size}", f"Horizon-Labs/multilingual-edu-classifier-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st, te = first(ours_f), json.load(open(bl_f)), json.load(open(st_f)), json.load(open(te_f))
ot = first(other_f) if other_f != "-" else None
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
BL = [("HuggingFaceFW/fineweb-edu-classifier", "109M", "Apache-2.0", "English; trained on the Llama-3 annotations used for the English test (in-domain)"),
      ("nvidia/nemocurator-fineweb-mixtral-edu-classifier", "109M", "NVIDIA open model licence", "English"),
      ("nvidia/nemocurator-fineweb-nemotron-4-edu-classifier", "109M", "NVIDIA open model licence", "English"),
      ("hotchpotch/fineweb-2-edu-japanese-classifier", "279M", "MIT", "Japanese (XLM-R)"),
      ("lapa-llm/fineweb-nemotron-edu-score", "279M", "MIT", "Ukrainian (XLM-R)")]
link = lambda m: f"[{m}](https://huggingface.co/{m})"
row = lambda name, p, lic, r, note: (f"| {name} ({p}) | {lic} | {f3(r['fw2_gemma']['spearman'])} | {f3(r['fw2_gemma'].get('lang_mean_spearman'))} | "
                                      f"{f3(r['en_llama3']['spearman'])} | {f3(r['en_llama3']['f1_ge3'])} | {note} |")
rows = [row("**this model**", PARAMS[size], "Apache-2.0", me, "65+ languages")] + ([row(link(OTHER), PARAMS[other], "Apache-2.0", ot, "")] if ot else []) + \
       [row(link(m), p, lic, bl[m], note) for m, p, lic, note in BL if m in bl] + \
       [f"| Qwen3.8-27B, prompted (our teacher, raw scores before calibration; 27B) | Apache-2.0 | {f3(te['fw2']['spearman'])} | {f3(te['fw2']['lang_mean_spearman'])} | {f3(te['en']['spearman'])} | {f3(te['en']['f1_ge3'])} | for reference |"]
main = "\n".join(["| model | licence | 65 languages (FineWeb-2): Spearman | per-language mean Spearman | English (Llama-3 labels): Spearman | English: F1, educational (>= 3) | note |",
                  "|---|---|---|---|---|---|---|"] + rows)
bylang = me["fw2_gemma"]["by_lang"]; srt = sorted(bylang.items(), key=lambda kv: kv[1])
low = ", ".join(f"{l} {v:.2f}" for l, v in srt[:8]); high = ", ".join(f"{l} {v:.2f}" for l, v in srt[-8:][::-1])
FW = bl["HuggingFaceFW/fineweb-edu-classifier"]["en_llama3"]; d_en = me["en_llama3"]["spearman"] - FW["spearman"]
en_txt = (f"this model matches HuggingFaceFW/fineweb-edu-classifier (Spearman {me['en_llama3']['spearman']:.3f} vs {FW['spearman']:.3f})." if abs(d_en) < 0.005 else
          f"this model is {'ahead of' if d_en > 0 else 'behind'} HuggingFaceFW/fineweb-edu-classifier (Spearman {me['en_llama3']['spearman']:.3f} vs {FW['spearman']:.3f}).")
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB): scores differ from fp32 by {gq.get('mean_diff', 0):.3f} on average; "
        f"the educational / not decision (2.5) agrees for {100 * gq.get('agree', 0):.1f}% of {qs.get('n', 0)} test documents.") if gq else ""
langs = sorted({l for l in bylang})
card = f"""---
license: apache-2.0
language:
- multilingual
- en
- ru
- zh
- de
- ja
- es
- fr
- it
- pt
- pl
- nl
- id
- tr
- cs
- ko
- ar
- hu
- fa
- ro
- vi
- uk
- nb
- th
- el
- sv
- da
- fi
- bg
- sk
- hr
- hi
- lt
- he
- bn
- sl
- et
- ca
- lv
- ms
- sr
- ta
- ur
- mr
- te
- sw
- ka
- hy
- is
- mk
- kk
- az
- tl
- gl
- eu
- am
- ne
- gu
- kn
- ml
- pa
- si
- uz
- af
- km
- my
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb-edu-llama3-annotations
tags:
- fineweb-edu
- educational-quality
- data-filtering
- data-curation
- pretraining-data
- quality-classifier
- regression
- multilingual
- modernbert
- mmbert
- onnx
---

# Multilingual Educational-Quality Classifier ({size}, {PARAMS[size]})

Scores how **educational** a web page is, from 0 (none) to 5 (excellent, textbook-like), in **English and 65 other languages**, on the
same scale as [FineWeb-Edu](https://huggingface.co/HuggingFaceFW/fineweb-edu-classifier). Use it to filter or weight pretraining data,
for example FineWeb-2, the way FineWeb-Edu filtered English (`int_score >= 3`).
Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, with ONNX files included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- Drop-in for code written for `HuggingFaceFW/fineweb-edu-classifier`: one regression output; `int_score = round(clamp(score, 0, 5))`.
- It reads the first 512 tokens of a document. {onnx}

## Usage

```python
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

tok = AutoTokenizer.from_pretrained("{REPO}")
model = AutoModelForSequenceClassification.from_pretrained("{REPO}")
texts = ["La photosynthèse est le processus par lequel les plantes ...", "Buy cheap shoes now! Free shipping ..."]
inputs = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=512)
with torch.no_grad():
    score = model(**inputs).logits.squeeze(-1).float()
int_score = score.clamp(0, 5).round().int()
print(score.tolist(), int_score.tolist())
```

## Evaluation

- **65 languages**: 9,750 held-out FineWeb-2 documents (150 per language; never used for training) scored with the FineWeb-Edu rubric by an
  **independent judge**, [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) (a different model family from our teacher).
  Metric: Spearman rank correlation with the judge, over all documents and averaged per language.
- **English**: 6,000 documents from a held-out shard of [HuggingFaceFW/fineweb-edu-llama3-annotations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations)
  (Llama-3-70B scores); Spearman, and F1 of the educational decision (score >= 3, as FineWeb-Edu uses it).

{main}

- The table shows the released checkpoint. Means over {seeds}.
- The English classifiers rank multilingual documents poorly (Spearman about 0.3); this model is built for that case. On English, {en_txt}
  HuggingFaceFW/fineweb-edu-classifier was trained on the same Llama-3 annotations as the English test set (in-domain for it).
- Caveat: the multilingual reference is itself an LLM. The judge agrees with Llama-3 about as well as our teacher does (English Spearman
  0.73 vs 0.70), but the judge and our teacher agree with each other more (0.88), which favours this model somewhat on that test.
  No human-labelled multilingual educational-quality test set was available.
- Per language (Spearman with the judge): highest {high}; lowest {low}.

## Training

- **English**: {st['by_source']['llama3']:,} documents with their Llama-3-70B scores from
  [HuggingFaceFW/fineweb-edu-llama3-annotations](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu-llama3-annotations) (ODC-BY).
- **65 other languages**: {st['by_source']['fineweb2']:,} documents from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) (ODC-BY),
  about 2,000 per language, scored by Qwen3.8-27B (Apache-2.0) with the same rubric (expected score from its answer probabilities). Those
  scores were mapped onto the Llama-3 scale by isotonic regression fitted on {st['calibration']['n']:,} English documents scored by both,
  so that 3 means the same thing in every language.
- mmBERT-{size} with one regression output, mean squared error, max length 512 tokens; checkpoint chosen by Spearman on {st['val']:,}
  held-out documents. Code: `code/` in this repository.

## Limitations

- "Educational" here means the FineWeb-Edu rubric (useful for teaching at primary to grade-school level). It undervalues good content that is not
  school-like (fiction, news, forums, specialised research), and it can encode the judges' cultural biases. Filtering with it changes what
  a model trained on the data sees, so check that the result still covers what you need.
- Labels come from LLMs, not people, and the per-language agreement varies (see above). Lower-resource languages and scripts are weaker.
- Only the first 512 tokens are read.
"""
open(out, "w").write(card)
print("written", out, len(card))
