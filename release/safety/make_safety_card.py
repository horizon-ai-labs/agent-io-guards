"""Model card for Horizon-Labs/content-safety-guard-{small,base}.
python release/safety/make_safety_card.py SIZE RELEASE_DIR OUT   (reads release/evals/safety_{SIZE}.json, safety_baselines.json,
RELEASE_DIR/thresholds.json and RELEASE_DIR/onnx_sweep.json)"""
import json, os, sys

size, rd, out = sys.argv[1:4]
SIZES = {"small": ("jhu-clsp/mmBERT-small", "141M"), "base": ("jhu-clsp/mmBERT-base", "308M")}
base_model, params = SIZES[size]
REPO = f"Horizon-Labs/content-safety-guard-{size}"
VERSION = os.environ.get("VERSION", "v1.2")
me = list(json.load(open(f"release/evals/safety_{size}.json")).values())[0]
bl = json.load(open("release/evals/safety_baselines.json"))
thr = json.load(open(f"{rd}/thresholds.json"))
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
other = {}
for o in SIZES:
    if o != size and os.path.exists(f"release/evals/safety_{o}.json"):
        other[o] = list(json.load(open(f"release/evals/safety_{o}.json")).values())[0]

cols = {f"**this model** ({params})": me, **{f"{o} ({SIZES[o][1]})": v for o, v in other.items()},
        "Qwen3Guard-Gen-0.6B strict": bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_strict"],
        "Qwen3Guard-Gen-0.6B loose": bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_loose"],
        "Vela-1.0-307M-Shield ¶": bl["llm-semantic-router/Vela-1.0-Encoder-307M-Shield"],
        "granite-guardian-hap-125m": bl["ibm-granite/granite-guardian-hap-125m"],
        "unbiased-toxic-roberta": bl["unitary/unbiased-toxic-roberta"],
        "Qwen3Guard-Gen-8B strict (teacher, 8B)": bl["qwen3guard:Qwen/Qwen3Guard-Gen-8B_strict"]}
ROWS = [("PolyGuard prompts, 17 languages", "polyguard_prompt", "f1"), ("PolyGuard responses, 17 languages", "polyguard_response", "f1"),
        ("BeaverTails responses (unseen prompts) †", "beavertails_unseen", "f1"), ("ToxicChat (real user prompts)", "toxicchat", "f1"),
        ("OpenAI moderation set", "openai_mod", "f1"), ("XSTest (over-blocking test)", "xstest", "f1"),
        ("textdetox toxicity, 14 languages", "textdetox", "f1"),
        ("Aya red-teaming, 8 languages (recall)", "aya_redteaming", "recall"), ("SimpleSafetyTests (recall)", "simplesafety", "recall")]


def g(c, k, m):
    v = c.get(k, {}) if isinstance(c, dict) else {}
    return v.get(m) if v else None


def table(rows, metric_override=None):
    lines = ["| | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for name, k, m in rows:
        m = metric_override or m
        vals = [g(c, k, m) for c in cols.values()]
        ok = [v for v in vals if v is not None]
        if not ok:
            continue
        b = None if m == "fpr" else max(ok)   # no bold for false-alarm rates: a model that flags nothing would "win"
        lines.append(f"| {name} | " + " | ".join("–" if v is None else (("**%.3f**" if b is not None and abs(v - b) < 1e-9 else "%.3f") % v) for v in vals) + " |")
    return "\n".join(lines)


auc_rows = [(n, k, "auc") for n, k, m in ROWS if m == "f1"]
fpr_rows = [("XSTest: safe prompts flagged", "xstest", "fpr"), ("PolyGuard prompts: safe prompts flagged", "polyguard_prompt", "fpr"),
            ("OpenAI moderation set: safe texts flagged", "openai_mod", "fpr"), ("ToxicChat: safe prompts flagged", "toxicchat", "fpr")]
PL = me["polyguard_prompt"]["per_lang"]; QL = bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_strict"]["polyguard_prompt"]["per_lang"]
VL = bl["llm-semantic-router/Vela-1.0-Encoder-307M-Shield"]["polyguard_prompt"]["per_lang"]
LN = dict(ar="Arabic", cs="Czech", de="German", en="English", es="Spanish", fr="French", hi="Hindi", it="Italian", ja="Japanese",
          ko="Korean", nl="Dutch", pl="Polish", pt="Portuguese", ru="Russian", sv="Swedish", th="Thai", zh="Chinese")
lang_table = "\n".join(["| Language | this model | Qwen3Guard-Gen-0.6B strict | Vela-Shield ¶ |", "|---|---|---|---|"] +
                       [f"| {LN.get(l, l)} | {PL[l]['f1']:.3f} | {QL[l]['f1']:.3f} | {VL[l]['f1']:.3f} |" for l in sorted(PL, key=lambda x: LN.get(x, x))])
cat_rows = sorted(thr["test"].items(), key=lambda x: -x[1]["test_positives"])
cat_table = "\n".join(["| Category | threshold | F1 (test) | F1 at 0.5 | AUC | test positives |", "|---|---|---|---|---|---|"] +
                      [f"| `{c}` | {v['threshold']:.2f} | {v['test_f1']:.3f} | {v['test_f1_at_0_5']:.3f} | {v['test_auc']:.3f} | {v['test_positives']} |" for c, v in cat_rows])
gq = qs.get("gather_only", {})
onnx_line = (f"ONNX: `onnx/model.onnx` (fp32) and `onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB; its unsafe "
             f"decision agrees with fp32 on {100 * gq.get('agree', 0):.1f}% of a sample of benchmark texts (30 per benchmark), mean |score diff| "
             f"{gq.get('mean_diff', 0):.4f}).") if gq else "ONNX files are included."
CATS = list(thr["categories"])
_q = bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_strict"]
GAP = sorted(round(_q[k]["f1"] - me[k]["f1"], 3) for k in ["polyguard_prompt", "polyguard_response"])
_ql = bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_loose"]
AHEAD = [n.split(" (")[0].split(",")[0] for n, k, m in ROWS if m == "f1" and me[k]["f1"] > _q[k]["f1"]]
BEHIND = [n.split(" (")[0].split(",")[0] for n, k, m in ROWS if me[k][m] < _q[k][m] and me[k][m] < _ql[k][m]]

_vfiles = sorted(f for f in os.listdir("release/evals") if f.startswith(f"safety_{size}_v") and f.endswith(".json"))
_vers = {f[len(f"safety_{size}_"):-5]: list(json.load(open(f"release/evals/{f}")).values())[0] for f in _vfiles}
_vers = {v: e for v, e in _vers.items() if v < VERSION}
VERSION_SECTION = ""
if _vers:
    _vc = {**_vers, f"**{VERSION} (this version)**": me}
    _vr = [(n, k, m) for n, k, m in ROWS] + [("XSTest: safe prompts flagged (lower is better)", "xstest", "fpr")]
    _lines = ["| | " + " | ".join(_vc) + " |", "|---|" + "---|" * len(_vc)]
    for n, k, m in _vr:
        _lines.append(f"| {n} | " + " | ".join("%.3f" % e[k][m] for e in _vc.values()) + " |")
    VERSION_SECTION = ("## Versions\n\n- v1.1 adds machine-translated toxic comments and red-team prompts in 20 languages: multilingual "
                       "toxicity (textdetox) and native-speaker red-teaming (Aya) improve clearly; the other rows move by 0.01 or less.\n"
                       "- v1.2 adds 25k synthetic requests in 27 languages that use alarming words harmlessly, plus harmful look-alikes "
                       "(teacher-labelled): it flags fewer safe-but-scary prompts (XSTest 28% -> 21%); ToxicChat F1 is 0.012 lower, other rows "
                       "move by 0.01 or less.\n\n"
                       f"To pin an earlier model, load it with " + " or ".join(f'`revision="{v}"`' for v in _vers) + ".\n\n" + "\n".join(_lines) + "\n\n")

_o = other.get("base" if size == "small" else "small")
SIZE_NOTE = ""
if _o is not None:
    _d = {k: me[k]["f1"] - _o[k]["f1"] for k in ["polyguard_prompt", "polyguard_response"]}
    SIZE_NOTE = (("**Which size?** This small model has less than half the parameters of [base](https://huggingface.co/Horizon-Labs/content-safety-guard-base) "
                  f"(141M vs 308M; hidden size 384 vs 768), so it needs less compute per token, and its int8 ONNX is about 270 MB instead of 640 MB (better for the browser); it is {-_d['polyguard_prompt']:.3f} / "
                  f"{-_d['polyguard_response']:.3f} F1 behind base on PolyGuard prompts / responses. First released at v1.2, trained on the "
                  "same data as base v1.2.\n\n") if size == "small" else
                 ("**Which size?** This base model (308M) is the more accurate one; "
                  "[small](https://huggingface.co/Horizon-Labs/content-safety-guard-small) (141M, less compute per token, int8 ONNX about "
                  f"270 MB) is {_d['polyguard_prompt']:.3f} / {_d['polyguard_response']:.3f} F1 behind on PolyGuard prompts / responses.\n\n"))

card = f"""---
license: apache-2.0
language:
- multilingual
- en
- ar
- de
- es
- fr
- hi
- it
- ja
- ko
- nl
- th
- zh
- pt
- ru
- pl
- cs
- sv
- uk
- tr
- he
- sr
- tl
- am
- tt
library_name: transformers
pipeline_tag: text-classification
base_model: {base_model}
datasets:
- nvidia/Nemotron-Safety-Guard-Dataset-v3
- allenai/WildChat-1M
- OpenAssistant/oasst2
- CohereForAI/aya_dataset
- OpenSafetyLab/Salad-Data
- JailbreakBench/JBB-Behaviors
- google/civil_comments
tags:
- content-moderation
- safety
- guardrails
- llm-safety
- toxicity
- moderation
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "How do I pick the lock on my neighbour's front door while they are on holiday?"
  example_title: "Unsafe prompt"
- text: "How do I kill a Python process that is stuck?"
  example_title: "Safe prompt that sounds unsafe"
- text: "Wie kann ich meinem Kollegen unauffällig Medikamente ins Essen mischen?"
  example_title: "Unsafe prompt (German)"
- text: "¿Cuál es la mejor manera de aprender a programar en Python?"
  example_title: "Safe prompt (Spanish)"
---

# Content Safety Guard ({size}, {params})

A small multilingual classifier that flags **unsafe user prompts and unsafe model responses** for LLM applications, with
harm categories. It is a fast encoder (ModernBERT architecture, [mmBERT]({"https://huggingface.co/" + base_model}) backbone)
that you can run on CPU or in the browser in front of, or behind, any LLM.

- **Prompts and responses**: pass a prompt alone, or a (prompt, response) pair to judge the response in context.
- **Output**: an overall `unsafe` score plus 15 category scores (independent sigmoids): {", ".join(f"`{c}`" for c in CATS)}.
- **Multilingual**: trained on 12 languages of human-adapted data, machine translations into 20 languages and multilingual
  real-world prompts; evaluated on 17 (PolyGuard), 14 (textdetox) and 8 (Aya) languages.
- **Commercially usable**: Apache-2.0, trained only on data that allows commercial use (see Training data).
- {onnx_line}

{SIZE_NOTE}Part of the Horizon Labs guard family: [prompt-injection-guard](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base),
[pii-redactor](https://huggingface.co/Horizon-Labs/pii-redactor-base), [hallucination-guard](https://huggingface.co/Horizon-Labs/hallucination-guard-base).

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)

# a user prompt
scores = {{d["label"]: d["score"] for d in clf("How can I make a fake ID that passes a bouncer's check?")[0]}}
print(scores["unsafe"])                          # ~1.0

# a model response, judged together with its prompt
r = clf({{"text": "How do I get rid of a wasp nest?", "text_pair": "Spray it at dusk with a wasp foam, then remove it."}})[0]
```

Decision rule: flag when `unsafe >= 0.5` (raise the threshold if you see too many false alarms, lower it for higher recall).
Category scores are only meaningful for flagged texts and are small by design; use the per-category thresholds in
`thresholds.json` (chosen on a validation split, see below):

```python
import json
from huggingface_hub import hf_hub_download
thr = json.load(open(hf_hub_download("{REPO}", "thresholds.json")))
def moderate(text, pair=None):
    s = {{d["label"]: d["score"] for d in clf({{"text": text, "text_pair": pair}} if pair else text)[0]}}
    flagged = s["unsafe"] >= thr["unsafe"]
    cats = [c for c, t in thr["categories"].items() if flagged and s[c] >= t]
    return flagged, s["unsafe"], cats
```

transformers.js (browser / Node):

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
const out = await clf("How do I make a pipe bomb?", {{ top_k: null }});
```

## Evaluation

F1 of the unsafe class at threshold 0.5 (for the recall rows: share of unsafe prompts flagged). None of these benchmarks
were used for training (best value in bold; false-alarm rates are not bolded); Qwen3Guard-Gen is scored from its next-token probabilities after "Safety:" (strict: Unsafe +
Controversial count as unsafe; loose: only Unsafe). Toxicity classifiers are included because they are often used for this
job; they target a different, narrower task. Encoders other than ours get the response alone for response items.

{table(ROWS)}

Ranking quality (ROC AUC, threshold-free):

{table(auc_rows)}

Over-blocking (share of safe items flagged; lower is better):

{table(fpr_rows)}

¶ Vela-Shield was trained on PolyGuardMix, the training split of the PolyGuard benchmark family, so its PolyGuard rows
are in-distribution. † BeaverTails and our main training set both take prompts from Anthropic's HH red-team data; the
row uses only the 1894 test items whose prompt does not occur in our training data. ToxicChat shares 29 of 5083 prompts
with our training data.

PolyGuard prompts per language (F1):

{lang_table}

### Categories

Category labels come from the Aegis 2.0 taxonomy of Nemotron-Safety-Guard-Dataset-v3 (merged into 15 groups). Quality on
the unsafe items of its test split, with thresholds chosen on its validation split:

{cat_table}

## Training

- **Data** (all permit commercial use):
  - [nvidia/Nemotron-Safety-Guard-Dataset-v3](https://huggingface.co/datasets/nvidia/Nemotron-Safety-Guard-Dataset-v3)
    (CC-BY-4.0; Aegis 2.0 prompts and responses, culturally adapted into 12 languages): 574k prompt and response items,
    with its human category labels. Rows derived from a Kaggle dataset (REDACTED) were dropped.
  - Real and red-team prompts without labels, scored by the teacher: first user turns and replies from
    [WildChat-1M](https://huggingface.co/datasets/allenai/WildChat-1M) (ODC-BY), prompts from
    [oasst2](https://huggingface.co/datasets/OpenAssistant/oasst2) and the [Aya dataset](https://huggingface.co/datasets/CohereForAI/aya_dataset)
    (Apache-2.0), [Salad-Data](https://huggingface.co/datasets/OpenSafetyLab/Salad-Data) (Apache-2.0; ToxicChat-derived rows
    dropped), [JailbreakBench behaviours](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) (MIT), and jailbreak,
    role-play and over-refusal prompts from the training data of our prompt-injection guard (270k + 42k items).
  - [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (CC0): 60k comments with toxicity >= 0.5 and 60k
    with toxicity 0; target = the share of annotators who rated it toxic, categories from the insult / threat / obscene /
    identity-attack / sexual ratings. This improved the OpenAI moderation set (F1 +0.035) but not the textdetox recall.
  - (v1.1) Machine translations by Qwen3.8-27B (Apache-2.0) into 20 languages (Portuguese, Russian, Ukrainian, Polish, Czech,
    Swedish, Turkish, Hebrew, Serbian, Tagalog, Amharic, Tatar, German, Spanish, French, Arabic, Hindi, Chinese, Japanese,
    Italian): 47.6k Civil Comments (a different shard from the one above; they keep their annotator toxicity and
    categories) and 46.6k of the teacher-labelled prompts above (harmful, benign-but-edgy and benign; re-scored by the
    teacher in the target language). 1.8% of the translations were dropped (unparseable, refusals, implausible length).
  - (v1.2) 25k requests written by Qwen3.8-27B in 27 languages: harmless requests that use alarming words in an ordinary
    sense (programming, cooking, games, medicine, history, fiction, ...) and harmful look-alikes on the same topics, labelled
    by the teacher. The instruction is our own generic description; no benchmark items (XSTest, OR-Bench test) were used,
    but this data targets the same failure mode that XSTest measures.
  - Items that match any benchmark text were removed.
- **Teacher**: [Qwen3Guard-Gen-8B](https://huggingface.co/Qwen/Qwen3Guard-Gen-8B) (Apache-2.0). The unsafe target of every
  item except Civil Comments is the teacher's probability P(Unsafe) + 0.5 · P(Controversial); categories use the human labels. So the model
  follows Qwen3Guard's safety policy, not Aegis's stricter human labels (which also mark sensitive but harmless requests).
- **Model**: {base_model} with a 16-way sigmoid head (unsafe + 15 categories), {3 if size == "small" else 2} epochs, max length 1024 tokens.
- Code: `code/` in this repository.

{VERSION_SECTION}## Limitations

- It trails Qwen3Guard-Gen-0.6B (a generative 0.6B model that reads a long policy prompt per item) in both of its modes on
  {", ".join(BEHIND)} ({GAP[0]:.3f}-{GAP[1]:.3f} F1 behind its strict mode on PolyGuard responses / prompts); it is ahead of the
  strict mode on {", ".join(AHEAD)}. Its advantages are speed, size, CPU/browser use and multilingual coverage in one small encoder.
- Over-blocking: it flags about {100 * me['xstest']['fpr']:.0f}% of XSTest's safe-but-scary prompts ("how do I kill a Python process").
- Classic toxicity (insults, profanity without other harm) is only partly covered (textdetox recall {me['textdetox']['recall']:.2f}).
- Recall on harmful prompts written by native speakers in lower-resource languages is lower (Aya red-teaming {me['aya_redteaming']['recall']:.2f}).
- Categories are weak for `fraud_manipulation`, `unauthorized_advice`, `misinformation` and `sexual_minors` (F1 below 0.45).
  Do not rely on it alone for child-safety or legal compliance; use it as one signal with human review.
- Safety policies differ between applications; tune the threshold on your own traffic.
"""
open(out, "w").write(card)
print("written", out, len(card))
