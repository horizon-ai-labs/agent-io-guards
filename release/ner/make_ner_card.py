"""Model card for Horizon-Labs/multilingual-ner-{small,base}.
python release/ner/make_ner_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json TEACHER.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, te_f, seeds, st_f, out = sys.argv[1:10]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-ner-{size}", f"Horizon-Labs/multilingual-ner-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, te, st = first(ours_f), json.load(open(bl_f)), json.load(open(te_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
oc = json.load(open(f"{rd}/onnx_check.json")) if os.path.exists(f"{rd}/onnx_check.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
BL = [("dslim/bert-base-NER", "108M", "MIT", "English (CoNLL-2003)"),
      ("Davlan/xlm-roberta-base-ner-hrl", "278M", "AFL-3.0", "10 languages (incl. CoNLL-2003); PER/ORG/LOC/DATE"),
      ("Davlan/bert-base-multilingual-cased-ner-hrl", "177M", "AFL-3.0", "10 languages; PER/ORG/LOC/DATE"),
      ("Babelscape/wikineural-multilingual-ner", "177M", "CC-BY-NC-SA-4.0", "9 languages; trained on WikiNEuRal (in-domain)")]
link = lambda m: f"[{m}](https://huggingface.co/{m})"
row = lambda name, lic, r, note: f"| {name} | {lic} | {f3(r['_conll'])} | {f3(r['_wikineural'])} | {f3(r['_wikiann'])} | {note} |"
rows = [row(f"**this model** ({PARAMS[size]})", "Apache-2.0", me, "47 training languages")] + \
       ([row(f"{link(OTHER)} ({PARAMS[other]})", "Apache-2.0", ot, "")] if ot else []) + \
       [row(f"{link(m)} ({p})", lic, bl[m], note) for m, p, lic, note in BL] + \
       [row("Qwen3.8-27B (our teacher, prompted; 300 sentences per set)", "Apache-2.0", te, "27B LLM, for reference")]
main = "\n".join(["| model | licence | CoNLL-2003 (English) | WikiNEuRal (9 languages) | WikiANN (31 languages, PER/ORG/LOC) | note |",
                  "|---|---|---|---|---|---|"] + rows)
ahead = {k: [m.split("/")[1] for m, *_ in BL if bl[m][k] > me[k]] for k in ["_conll", "_wikineural", "_wikiann"]}
LN = dict(ar="Arabic", bg="Bulgarian", bn="Bengali", cs="Czech", da="Danish", de="German", el="Greek", en="English", es="Spanish", fa="Persian",
          fi="Finnish", fr="French", he="Hebrew", hi="Hindi", hu="Hungarian", id="Indonesian", it="Italian", ja="Japanese", ko="Korean", nl="Dutch",
          pl="Polish", pt="Portuguese", ro="Romanian", ru="Russian", sv="Swedish", sw="Swahili", th="Thai", tr="Turkish", uk="Ukrainian",
          vi="Vietnamese", zh="Chinese")
D = "Davlan/xlm-roberta-base-ner-hrl"
wa = sorted(k for k in me if k.startswith("wikiann_"))
per = "\n".join(["| WikiANN language | this model | xlm-roberta-base-ner-hrl |", "|---|---|---|"] +
                [f"| {LN.get(k[8:], k[8:])} | {f3(me[k]['f1'])} | {f3(bl[D][k]['f1'])} |" for k in wa])
wn = "\n".join(["| WikiNEuRal language | this model | xlm-roberta-base-ner-hrl | wikineural-multilingual-ner (in-domain) |", "|---|---|---|---|"] +
               [f"| {LN.get(k[11:], k[11:])} | {f3(me[k]['f1'])} | {f3(bl[D][k]['f1'])} | {f3(bl['Babelscape/wikineural-multilingual-ner'][k]['f1'])} |"
                for k in sorted(x for x in me if x.startswith("wikineural_"))])
q = oc.get("model_quantized.onnx", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {q.get('mb', 0):.0f} MB) predicts the same tag as fp32 for "
        f"{100 * q.get('token_agreement', 0):.2f}% of tokens on held-out text.") if q else ""
ent = st["entities"]

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
- bg
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
- yo
- ca
- sr
- hr
- sk
- sl
- lt
- lv
- et
library_name: transformers
pipeline_tag: token-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- ner
- named-entity-recognition
- token-classification
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Angela Merkel met Emmanuel Macron in Paris to discuss the European Union."
  example_title: "English"
- text: "El Real Madrid fichó al delantero brasileño Vinícius Júnior."
  example_title: "Spanish"
- text: "李华在北京大学学习中文。"
  example_title: "Chinese"
- text: "Die Deutsche Bahn eröffnet einen neuen Bahnhof in Stuttgart."
  example_title: "German"
---

# Multilingual NER ({size}, {PARAMS[size]}): PER, ORG, LOC, MISC

A named-entity recognizer for **people (PER), organizations (ORG), locations (LOC) and other names (MISC)** in 47
languages, using the CoNLL label set (IOB tags: `B-PER`, `I-PER`, ...). It is a multilingual, Apache-2.0
alternative to [dslim/bert-base-NER](https://huggingface.co/dslim/bert-base-NER) built on
[mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}). It was trained on openly licensed web text labelled by an
Apache-2.0 LLM, so it has no non-commercial dataset restrictions. ONNX files for CPU and the browser (transformers.js) are
included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- MISC covers nationalities and languages ("German"), events, products, works and laws.
- Works on raw text with the `token-classification` pipeline and `aggregation_strategy="simple"`, in any script
  (no word segmentation needed for Chinese, Japanese or Thai).
- {onnx}

## Usage

```python
from transformers import pipeline

ner = pipeline("token-classification", model="{REPO}", aggregation_strategy="simple")
ner("Angela Merkel met Emmanuel Macron in Paris to discuss the European Union.")
# [{{'entity_group': 'PER', 'word': 'Angela Merkel', ...}}, {{'entity_group': 'PER', 'word': 'Emmanuel Macron', ...}},
#  {{'entity_group': 'LOC', 'word': 'Paris', ...}}, {{'entity_group': 'ORG', 'word': 'European Union', ...}}]
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const ner = await pipeline("token-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await ner("Die Deutsche Bahn eröffnet einen neuen Bahnhof in Stuttgart.", {{ aggregation_strategy: "simple" }}));
```

## Evaluation

Entity-level micro F1 (exact span and type) on public NER test sets, used only for evaluation. Every model reads the
same sentences as plain text, through the same script (`code/`). Labels a set does not annotate are ignored (WikiANN
has no MISC), and other types such as DATE are mapped to O.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- Models ahead of this one: CoNLL-2003: {", ".join(ahead["_conll"]) or "none"}; WikiNEuRal: {", ".join(ahead["_wikineural"]) or "none"};
  WikiANN: {", ".join(ahead["_wikiann"]) or "none"}. dslim/bert-base-NER and the Davlan models were trained on CoNLL-2003
  itself, and wikineural-multilingual-ner on WikiNEuRal, so those comparisons are in-domain for them. Our model has seen
  no data from these benchmarks.
- WikiANN annotations are automatic (derived from Wikipedia links), and their boundary conventions differ from CoNLL's, so
  every model scores lower there.

Per language:

{wn}

{per}

## Training

- **Text**: {st['n_train']:,} web passages (2-6 sentences) in 47 languages from
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) (ODC-BY).
- **Labels**: Qwen3.8-27B (Apache-2.0) listed the named entities of each passage with CoNLL-style guidelines and examples
  (`code/ner/teacher_ner.py`). We chose that prompt over a simpler one on the benchmarks' *validation* splits. The entity
  strings were aligned back to character spans ({ent.get('PER', 0):,} PER, {ent.get('ORG', 0):,} ORG, {ent.get('LOC', 0):,} LOC,
  {ent.get('MISC', 0):,} MISC).
- **Model**: mmBERT-{size} token classifier (IOB tags), 3 epochs, max 384 tokens per window.
- Code: `code/` in this repository.

## Limitations

- The labels come from an LLM, so the model inherits its boundary and type decisions, which sometimes differ from a
  given dataset's guidelines (e.g. whether titles, articles or inflections are part of a name).
- Lower accuracy in languages and domains that are rare in web text (see the per-language tables).
- Only these four types. For personal data such as emails, phone numbers and IDs, use our
  [pii-redactor](https://huggingface.co/Horizon-Labs/pii-redactor-small).
"""
open(out, "w").write(card)
print("written", out, len(card))
