"""Model card for Horizon-Labs/pii-redactor-{small,base}.

python release/pii/make_pii_card.py --size small --ext <pii_eval.json with ours + competitors> --ours_key <key of our model>
    --indist <eval_results.json from train_tok.py> [--other_key <key of the other size>] --out release/pii/small/README.md
"""
import argparse, json

ap = argparse.ArgumentParser()
ap.add_argument("--size", required=True)
ap.add_argument("--ext", required=True)
ap.add_argument("--ours_key", required=True)
ap.add_argument("--other_key", default="")
ap.add_argument("--indist", required=True)
ap.add_argument("--out", required=True)
args = ap.parse_args()

SIZES = {"small": ("jhu-clsp/mmBERT-small", "141M"), "base": ("jhu-clsp/mmBERT-base", "308M")}
base_model, params = SIZES[args.size]
REPO = f"Horizon-Labs/pii-redactor-{args.size}"
ext = json.load(open(args.ext))
indist = json.load(open(args.indist))
NAMES = {"openai/privacy-filter": "OpenAI Privacy Filter (1.5B)", "gravitee-io/bert-small-pii-detection": "gravitee bert-small",
         "OpenMed/OpenMed-PII-SuperClinical-Small-44M-v1": "OpenMed PII Small 44M"}
cols = {"**this model**": ext[args.ours_key]}
if args.other_key:
    cols["base (308M)" if args.size == "small" else "small (141M)"] = ext[args.other_key]
for k, n in NAMES.items():
    if k in ext and "error" not in ext[k]:
        cols[n] = ext[k]
SETS = [("redactionbench", "RedactionBench (real-world style forms, letters, syllabi; 'mandatory' spans)"),
        ("privacy_bench", "TonicAI Privacy-Bench (corporate email threads)"),
        ("tab_echr", "TAB: ECHR court judgments (DIRECT identifiers)"),
        ("ru_pii_benchmark", "Russian PII benchmark (redmadrobot)"),
        ("ours_synthetic_pii", "Secrets, code, configs, chats in 36 languages (our held-out synthetic set) *")]


def row(metric):
    lines = ["| Benchmark | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for s, d in SETS:
        vals = [c.get(s, {}).get(metric) for c in cols.values()]
        b = max(v for v in vals if v is not None)
        lines.append(f"| {d} | " + " | ".join(("**%.3f**" if v == b else "%.3f") % v if v is not None else "–" for v in vals) + " |")
    return "\n".join(lines)


def rp_table():
    lines = ["| Benchmark | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for s, d in SETS:
        cells = []
        for c in cols.values():
            r = c.get(s)
            cells.append(f"{r['redact_recall']:.2f} / {r['redact_precision']:.2f}" if r else "–")
        lines.append(f"| {d} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


ind_lines = ["| Held-out set | entity F1 (exact span + type) | redaction recall | redaction precision |", "|---|---|---|---|"]
for k, desc in [("openpii", "OpenPII validation (30 languages)"), ("nemotron", "Nemotron-PII test (en)"), ("gretel", "Gretel PII test (en)")]:
    if k in indist:
        r = indist[k]
        ind_lines.append(f"| {desc} | {r['entity_f1']:.3f} | {r['redact_recall']:.3f} | {r['redact_precision']:.3f} |")
labels = sorted({l for v in indist.values() for l in v.get("per_label", {})})
per_label = indist.get("nemotron", {}).get("per_label", {})

card = f"""---
license: apache-2.0
language:
- multilingual
- en
- de
- fr
- es
- it
- nl
- pt
- pl
- cs
- sk
- sl
- hr
- sr
- bg
- ro
- hu
- el
- da
- sv
- fi
- et
- lv
- lt
- ru
- uk
- tr
- ar
- hi
- zh
- ja
- ko
- vi
- id
- ms
- tl
- th
library_name: transformers
pipeline_tag: token-classification
base_model: {base_model}
tags:
- pii
- pii-detection
- redaction
- anonymization
- privacy
- secrets-detection
- data-masking
- ner
- presidio
- modernbert
- mmbert
- onnx
- transformers.js
datasets:
- ai4privacy/pii-masking-openpii-1.5m
- nvidia/Nemotron-PII
- gretelai/gretel-pii-masking-en-v1
---

# PII Redactor ({args.size}, {params})

A small, fast, **multilingual PII and secrets detector** for redacting text before it reaches logs, LLM prompts,
training data or a vendor API. It tags {len(labels)} entity types: names, contact details, addresses, government IDs,
financial data, credentials and API keys, network and device identifiers, and more.

- **Open**: Apache-2.0, ungated, trained only on permissively licensed data.
- **Multilingual**: 30+ languages, Latin and non-Latin scripts.
- **Small**: {params} parameters, CPU-friendly. ONNX and transformers.js work in the browser and at the edge.
- **Measured honestly**: evaluated on four external benchmarks (plus one held-out set of our own) that no model below was trained on, against the most
  used open PII models. It is not the best on every benchmark; see the table.

Try it in the browser: [Horizon-Labs/pii-redactor demo](https://huggingface.co/spaces/Horizon-Labs/pii-redactor).

Part of [Agent I/O Guards](https://huggingface.co/collections/Horizon-Labs/agent-i-o-guards-6ab403c49494bc2b71ca7669),
alongside [Prompt Injection Guard](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base).

## Quick start

```python
from transformers import pipeline

ner = pipeline("token-classification", model="{REPO}", aggregation_strategy="simple")
ner("Hi, I'm Anna Müller. Mail anna.mueller@posteo.de, key sk-proj-9fQ2x7LmA1bC3dE4")
```

For redaction, use `redact.py` from this repo. It merges sub-word pieces, trims whitespace, and handles long
documents with overlapping windows:

```python
from huggingface_hub import hf_hub_download
import importlib.util, sys
spec = importlib.util.spec_from_file_location("redact", hf_hub_download("{REPO}", "redact.py"))
redact = importlib.util.module_from_spec(spec); spec.loader.exec_module(redact)

r = redact.PIIRedactor("{REPO}")
r.redact("Bonjour, je m'appelle Jean Dupont, j'habite 12 rue de la Paix, 75002 Paris.")
# "Bonjour, je m'appelle [PERSON], j'habite [STREET_ADDRESS], [POSTCODE] [LOCATION]."
```

### Presidio

```python
from presidio_analyzer import AnalyzerEngine
analyzer = AnalyzerEngine()
analyzer.registry.add_recognizer(redact.presidio_recognizer("{REPO}"))
analyzer.analyze(text="Call me at (415) 555-0132", language="en")
```

### LLM Guard (Anonymize scanner)

`llm_guard_conf.py` in this repo maps the labels to Presidio / LLM Guard entities. It is a permissive, multilingual
alternative to the default Anonymize models:

```python
spec = importlib.util.spec_from_file_location("conf", hf_hub_download("{REPO}", "llm_guard_conf.py"))
conf = importlib.util.module_from_spec(spec); spec.loader.exec_module(conf)

from llm_guard.vault import Vault
from llm_guard.input_scanners import Anonymize
scanner = Anonymize(Vault(), recognizer_conf=conf.horizon_pii_conf("{args.size}"), language="en")
scanner.scan("Hi, I'm Anna Müller, email anna.mueller@posteo.de, IBAN DE89370400440532013000.")[0]
# "Hi, I'm [REDACTED_PERSON_1], email [REDACTED_EMAIL_ADDRESS_1], IBAN [REDACTED_IBAN_CODE_1]."
```

## Entity types

{", ".join(f"`{l}`" for l in labels)}

`ORGANIZATION` means an employer or company linked to a person. `SECRET` covers API keys, tokens, private keys,
connection strings and session cookies. Gender and other sensitive attributes (religion, health conditions) are
**not** tagged in this version.

## Evaluation

### External benchmarks

None of the models below were trained on these benchmarks. Every model uses its own label set, so the comparison
ignores entity types. It measures what matters for redaction: **redaction recall**, the share of must-redact
characters that get masked, and **redaction precision**, the share of masked characters that are PII. All models ran
through the same `transformers` token-classification pipeline (`code/pii/evaluate_pii.py`), with at most 600 documents
per benchmark.

Redaction recall / precision:

{rp_table()}

Redaction F1 (harmonic mean of the two):

{row("redact_f1")}

Reading this table:
- RedactionBench marks a lot as mandatory (form field values, course codes, IDs), so every model's recall is low.
  Relative order is the useful signal.
- Privacy-Bench (email threads): gravitee and OpenMed catch more PII but mask much more non-PII.
  OpenAI's much larger model is the most precise.
- \* The last row is a held-out set we generated with Qwen3.8-27B: secrets in code, config files and logs, plus
  chat and email threads, in 36 languages. This model was **not** trained on any of that generator's output, but we
  built the set ourselves, so read it as supporting evidence only.
- TAB counts DIRECT identifiers in court judgments, such as names and case-application numbers. Case numbers are not a
  type any of these models were trained for.

### In-distribution held-out sets

These come from the same generators as the training data, so they overstate real-world quality.

{chr(10).join(ind_lines)}

## Limitations

- Most training text is synthetic (OpenPII, Nemotron-PII, Gretel, plus Qwen3.8-generated documents in later versions).
  Real documents are messier, so expect lower recall on unusual formats, and review before relying on it for compliance.
- It is not a guarantee of anonymization. Quasi-identifiers (job title plus town plus age) and free-text descriptions
  can still identify people.
- Dates are tagged whether or not they are personal, which over-redacts public dates.
- Languages outside the training set, and long numeric strings with unusual grouping, are weaker. For example, a
  space-separated 16-digit card number can be only partly masked, and a bare CVV next to it can be missed.

## Training

- Backbone: [{base_model}](https://huggingface.co/{base_model}) (MIT), BIO token classification, max length 512
  with stride windows, bf16, 1 epoch.
- Data (about 400k documents): [OpenPII 1.5M](https://huggingface.co/datasets/ai4privacy/pii-masking-openpii-1.5m)
  (CC-BY-4.0, ai4privacy; language-balanced 256k sample), [Nemotron-PII](https://huggingface.co/datasets/nvidia/Nemotron-PII)
  (CC-BY-4.0, NVIDIA), and [Gretel PII masking EN v1](https://huggingface.co/datasets/gretelai/gretel-pii-masking-en-v1)
  (Apache-2.0). Their label sets were mapped to one Presidio-aligned taxonomy (`code/pii/build_pii_v0.py`).
- Attribution: this model is trained on CC-BY-4.0 data from ai4privacy and NVIDIA. Please keep this notice when
  redistributing derivatives.

## Citation

```bibtex
@misc{{horizonlabs2026piiredactor,
  title  = {{PII Redactor: small multilingual PII and secrets detection}},
  author = {{Horizon Labs}},
  year   = {{2026}},
  url    = {{https://huggingface.co/{REPO}}}
}}
```
"""
open(args.out, "w").write(card)
print(rp_table())
