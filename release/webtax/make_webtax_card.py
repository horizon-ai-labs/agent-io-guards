"""Model card for Horizon-Labs/multilingual-web-topic-format-{small,base}.
python release/webtax/make_webtax_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys
size, rd, ours_f, other_f, bl_f, seeds, st_f, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-web-topic-format-{size}", f"Horizon-Labs/multilingual-web-topic-format-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st = first(ours_f), json.load(open(bl_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
link = lambda m: f"[{m}](https://huggingface.co/{m})"
row = lambda name, r, note: (f"| {name} | {f3(r['fw2_judged']['topic']['acc'])} | {f3(r['fw2_judged']['format']['acc'])} | {f3(r['fw2_judged']['topic']['macro_f1'])} | "
                              f"{f3(r['en_llama405b']['topic']['acc'])} | {f3(r['en_llama405b']['format']['acc'])} | {note} |")
WO, T = "WebOrganizer (TopicClassifier-NoURL + FormatClassifier-NoURL)", "Qwen3.8-27B (teacher, prompted)"
rows = [row(f"**this model** ({PARAMS[size]})", me, "66 languages, topic + format in one model")] + ([row(f"{link(OTHER)} ({PARAMS[other]})", ot, "")] if ot else []) + \
       [row("[WebOrganizer/TopicClassifier-NoURL](https://huggingface.co/WebOrganizer/TopicClassifier-NoURL) + [FormatClassifier-NoURL](https://huggingface.co/WebOrganizer/FormatClassifier-NoURL) (2 x 140M)",
            bl[WO], "English; trained on the Llama-3.1-405B labels of the English test (in-domain)"),
        row("Qwen3.8-27B, prompted (our teacher, 27B)", bl[T], "for reference")]
main = "\n".join(["| model | 65 languages: topic accuracy | 65 languages: format accuracy | 65 languages: topic macro-F1 | English: topic accuracy | English: format accuracy | note |",
                  "|---|---|---|---|---|---|---|"] + rows)
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB) gives the same 0.5 decision as fp32 on all 48 outputs for {100 * gq.get('agree', 0):.1f}% "
        f"of {qs.get('n', 0)} test pages.") if gq else ""
langs = ["en", "ru", "zh", "de", "ja", "es", "fr", "it", "pt", "pl", "nl", "id", "tr", "cs", "ko", "ar", "hu", "fa", "ro", "vi", "uk", "nb", "th", "el", "sv", "da",
         "fi", "bg", "sk", "hr", "hi", "lt", "he", "bn", "sl", "et", "ca", "lv", "ms", "sr", "ta", "ur", "mr", "te", "sw", "ka", "hy", "is", "mk", "kk", "az", "tl",
         "gl", "eu", "am", "ne", "gu", "kn", "ml", "pa", "si", "uz", "af", "km", "my"]
card = f"""---
license: apache-2.0
language:
- multilingual
""" + "".join(f"- {l}\n" for l in langs) + f"""library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
tags:
- topic-classification
- domain-classification
- web-classification
- weborganizer
- data-curation
- pretraining-data
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
---

# Multilingual Web Topic & Format Classifier ({size}, {PARAMS[size]})

Labels a web page with its **topic** and its **format**, in English and 65 other languages, using the 24 + 24 categories of
[WebOrganizer](https://github.com/CodeCreator/WebOrganizer) (Wettig et al., 2025):

- **topic**: Adult Content, Art & Design, Software Development, Crime & Law, Education & Jobs, Electronics & Hardware, Entertainment, Social Life,
  Fashion & Beauty, Finance & Business, Food & Dining, Games, Health, History & Geography, Home & Hobbies, Industrial, Literature, Politics,
  Religion, Science, Math & Technology, Software, Sports & Fitness, Transportation, Travel & Tourism;
- **format**: Academic Writing, Content Listing, Creative Writing, Customer Support Page, Discussion Forum / Comment Section, FAQs, Incomplete
  Content, Knowledge Article, Legal Notices, Listicle, News Article, Nonfiction Writing, Organizational About Page, Organizational Announcement,
  Personal About Page, Personal Blog, Product Page, Q&A Forum, Spam / Ads, Structured Data, Technical Writing, Transcript / Interview,
  Tutorial / How-To Guide, User Reviews.

Use it to analyse, filter or rebalance multilingual pretraining data (for example FineWeb-2) by domain, as WebOrganizer did for English.
One model returns both: 48 sigmoid outputs, `topic_*` and `format_*`; take the highest of each group.
Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, ONNX included. It reads the first 512 tokens; no URL needed.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""} {onnx}

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)
s = {{d["label"]: d["score"] for d in clf("Rezept: Für den Teig 500 g Mehl, 2 Eier und eine Prise Salz verrühren ...")[0]}}
topic = max((k for k in s if k.startswith("topic_")), key=s.get)
fmt = max((k for k in s if k.startswith("format_")), key=s.get)
print(topic, fmt)   # topic_food_dining format_tutorial_how_to_guide
```

## Evaluation

- **65 languages**: 9,750 held-out FineWeb-2 pages (150 per language, never used for training) labelled with WebOrganizer's prompts by an
  independent judge, [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it) (a different model family from our teacher).
- **English**: 3,000 pages of the test split of WebOrganizer's own annotations by Llama-3.1-405B
  ([topics](https://huggingface.co/datasets/WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8), [formats](https://huggingface.co/datasets/WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8));
  used only for evaluation.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- WebOrganizer's classifiers are English models trained on the same Llama-3.1-405B annotations as the English test, so they lead there; on
  other languages they fall to about 0.4 (topic) and 0.25 (format) agreement with the judge.
- Caveat: the multilingual reference is an LLM. On English, the judge agrees with Llama-3.1-405B on 84% of topics and 71% of formats, and our
  teacher on 86% / 74%, but the judge and teacher agree with each other more, which favours this model on that test. Formats are more
  ambiguous than topics for every annotator.

## Training

- **Pages**: {st['train'] + st['val']:,} web pages, about 2,000 per language from [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2)
  and 22,000 English pages from FineWeb (ODC-BY).
- **Labels**: Qwen3.8-27B (Apache-2.0), prompted with WebOrganizer's topic and format instructions (Apache-2.0; without URL or demonstrations),
  gave a probability for each category; the model learns these as soft labels. The Llama-3.1-405B annotations were not used for training.
- mmBERT-{size}, max length 512 tokens; checkpoint chosen on {st['val']:,} held-out pages. Code: `code/` in this repository.

## Limitations

- Labels come from LLMs, not people, and many pages fit several categories (WebOrganizer itself reports format as the harder taxonomy).
- Only the text is used; WebOrganizer's URL-aware classifiers can use URL hints that this model does not see.
- Rare categories in some languages (e.g. Transcript / Interview, Legal Notices) have little training data.
"""
open(out, "w").write(card); print("written", out, len(card))
