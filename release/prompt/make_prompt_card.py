"""Model card for Horizon-Labs/multilingual-prompt-task-complexity-{small,base}.
python release/prompt/make_prompt_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys
size, rd, ours_f, other_f, bl_f, seeds, st_f, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-prompt-task-complexity-{size}", f"Horizon-Labs/multilingual-prompt-task-complexity-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st = first(ours_f), json.load(open(bl_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
D = ["creativity", "reasoning", "contextual_knowledge", "domain_knowledge", "constraints", "complexity"]
row = lambda name, r, note: (f"| {name} | {f3(r['task_acc'])} | {f3(r['task_macro_f1'])} | {f3(r['task_acc_en'])} | {f3(r['task_acc_lang_mean'])} | "
                              + " | ".join(f3(r.get('spearman_' + d)) for d in D) + f" | {f3(r['spearman_mean'])} | {note} |")
link = lambda m: f"[{m}](https://huggingface.co/{m})"
rows = [row(f"**this model** ({PARAMS[size]})", me, "multilingual")] + ([row(f"{link(OTHER)} ({PARAMS[other]})", ot, "")] if ot else []) + \
       [row(f"{link('nvidia/prompt-task-and-complexity-classifier')} (184M)", bl["nvidia"], "English; human-annotated training data, own definitions"),
        row("Qwen3.8-27B, prompted (our teacher, 27B)", bl["teacher"], "for reference")]
main = "\n".join(["| model | task accuracy | task macro-F1 | task accuracy, English | task accuracy, mean over languages | " + " | ".join(d.replace("_", " ") for d in D)
                  + " | dimensions: mean Spearman | note |", "|---" * (7 + len(D)) + "|"] + rows)
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB) gives the same 0.5 decisions as fp32 on all outputs for "
        f"{100 * gq.get('agree', 0):.1f}% of {qs.get('n', 0)} test prompts.") if gq else ""
card = f"""---
license: apache-2.0
language:
- multilingual
- en
- zh
- ru
- es
- fr
- de
- pt
- it
- ar
- tr
- ja
- ko
- vi
- id
- pl
- uk
- hu
- fa
- hi
- bn
- sw
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- allenai/WildChat-1M
- OpenAssistant/oasst2
- CohereLabs/aya_dataset
tags:
- prompt-classification
- task-classification
- prompt-complexity
- llm-routing
- model-routing
- data-curation
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Write a Python function that checks whether a string is a palindrome."
  example_title: "Code"
- text: "Résume ce texte en trois phrases : ..."
  example_title: "French summary"
- text: "Explain why the sky is blue to a five-year-old."
  example_title: "Open QA"
---

# Multilingual Prompt Task & Complexity Classifier ({size}, {PARAMS[size]})

Classifies a **user prompt** sent to an LLM, in English and 100+ other languages:

- **task type** (11, the taxonomy of NVIDIA's prompt classifier): open QA, closed QA, summarization, text generation, code generation, chatbot,
  classification, rewrite, brainstorming, extraction, other;
- **complexity** scores from 0 to 1: creativity, reasoning, contextual knowledge, domain knowledge, constraints, and overall complexity.

Typical uses: routing prompts to a cheaper or stronger model, analysing chat logs, balancing instruction-tuning data.
Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, ONNX for CPU and the browser (transformers.js) included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- 17 sigmoid outputs: `task_*` (take the highest as the task type) and the six dimensions. {onnx}
- The dimension scores are relative: most prompts score low (training averages: complexity {st['dim_means']['complexity']:.2f}, reasoning
  {st['dim_means']['reasoning']:.2f}, creativity {st['dim_means']['creativity']:.2f}). Compare prompts with each other, or pick routing thresholds from your own traffic.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)
s = {{d["label"]: d["score"] for d in clf("Écris une fonction Python qui vérifie si une chaîne est un palindrome.")[0]}}
task = max((k for k in s if k.startswith("task_")), key=s.get)
print(task, round(s["complexity"], 2), round(s["reasoning"], 2))
```

## Evaluation

There is no public human-labelled multilingual test set for this taxonomy. We held out 4,200 real user prompts from
[WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (200 in each of 21 languages, never used for training) and labelled them
with an **independent judge**, [google/gemma-4-31b-it](https://huggingface.co/google/gemma-4-31b-it), using the same questions as our teacher.
Task accuracy and macro-F1 compare the top task; the dimensions use Spearman rank correlation.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- Read the comparison with NVIDIA's classifier with care: it was trained on human annotations with its own guidelines, so part of its
  disagreement with our judge is about definitions, not errors; on English prompts the two agree on the task {bl['nvidia']['task_acc_en']:.0%} of the time. Outside
  English its agreement drops further, which is the gap this model fills.
- Caveat: the judge and our teacher are both LLMs and agree with each other closely (task {bl['teacher']['task_acc']:.0%}), which favours this model on this test.

## Training

- **Prompts**: {st['train'] + st['val']:,} real user prompts in {st['n_lang']} languages: first user messages from WildChat-1M (ODC-BY; only the
  prompt text was used, no metadata; conversations flagged toxic excluded), prompts from [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2)
  and [Aya](https://huggingface.co/datasets/CohereLabs/aya_dataset) (Apache-2.0). At most 3,000 per language (English 20,000, Chinese 8,000).
- **Labels**: Qwen3.8-27B (Apache-2.0) answered one multiple-choice question (task) and six 0-9 ratings per prompt; the model learns the
  answer probabilities (task) and expected ratings / 9 (dimensions) as soft targets. Checkpoint chosen on {st['val']:,} held-out prompts.
- mmBERT-{size}, max length 512 tokens. Code: `code/` in this repository.

## Limitations

- Labels come from an LLM, not people, and complexity ratings are subjective. Use the scores for ranking and routing, and validate thresholds on
  your own traffic.
- Only the first 512 tokens of a prompt are read; very long prompts with instructions at the end may be misclassified.
- Not a safety classifier: for harmful or injected prompts see our [content-safety-guard](https://huggingface.co/Horizon-Labs/content-safety-guard-small)
  and [prompt-injection-guard](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small).
"""
open(out, "w").write(card); print("written", out, len(card))
