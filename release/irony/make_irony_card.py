"""Model card for Horizon-Labs/multilingual-irony-sarcasm-{small,base}.
python release/irony/make_irony_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json|- BASELINES.json SEEDS_TEXT STATS.json OUT"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, seeds, st_f, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-irony-sarcasm-{size}", f"Horizon-Labs/multilingual-irony-sarcasm-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, bl, st = first(ours_f), json.load(open(bl_f)), json.load(open(st_f))
ot = first(other_f) if other_f != "-" else None
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
T = "Qwen/Qwen3.8-27B (teacher, prompted)"
BL = [("cardiffnlp/twitter-roberta-base-irony", "125M", "none given", "English tweets; trained on SemEval-2018 irony"),
      ("pysentimiento/bertweet-irony", "135M", "none given", "English tweets; trained on SemEval-2018 irony"),
      ("pysentimiento/robertuito-irony", "109M", "none given", "Spanish tweets"),
      ("rogelioplatt/distilbert-base-multilingual-cased-Sarcasmo_Esp::LABEL_1", "135M", "Apache-2.0", "Spanish sarcasm"),
      ("helinivan/english-sarcasm-detector::LABEL_1", "110M", "none given", "English sarcasm (news headlines)"),
      ("helinivan/multilingual-sarcasm-detector::LABEL_1", "178M", "none given", "multilingual sarcasm (news headlines)"),
      (T, "27B", "Apache-2.0", "our teacher, for reference")]
nm = lambda m: m.split("::")[0]
link = lambda m: nm(m) if " " in m else f"[{nm(m)}](https://huggingface.co/{nm(m)})"
TW = [("semeval18", "SemEval-2018 irony (EN)"), ("isarcasm", "iSarcasmEval (EN)"), ("ar_sarcasm", "ArSarcasm (Arabic)"), ("ko_sarcasm", "Korean sarcasm")]
mp_noctx = lambda r: (sum(r[k].get("auc_noctx", r[k]["auc"]) for k in r if k.startswith("mpico_")) / sum(k.startswith("mpico_") for k in r))
row = lambda name, p, lic, r, note, ctx: (f"| {name} ({p}) | {lic} | {f3(r['_mpico_mean_auc']) if ctx else '—'} | {f3(mp_noctx(r))} | "
                                          + " | ".join(f3(r[k]["auc"]) for k, _ in TW) + f" | {f3(r['_tweets_mean_auc'])} | {note} |")
tmp = dict(bl[T]); tmp.update({k: dict(v, auc_noctx=v.get("auc_noctx", v["auc"])) for k, v in bl[T].items() if k.startswith("mpico_")})
rows = [row("**this model**", PARAMS[size], "Apache-2.0", me, "multilingual; uses the post as context", True)] + \
       ([row(link(OTHER), PARAMS[other], "Apache-2.0", ot, "", True)] if ot else []) + \
       [row(link(m), p, lic, bl[m] if m != T else tmp, note, m == T) for m, p, lic, note in BL if m in bl]
main = "\n".join(["| model | licence | MultiPICo, 9 languages: AUC with the post | MultiPICo: AUC reply only | " + " | ".join(n + ": AUC" for _, n in TW)
                  + " | tweets: mean AUC | note |", "|---" * (5 + len(TW)) + "|---|"] + rows)
LN = dict(ar="Arabic", de="German", en="English", es="Spanish", fr="French", hi="Hindi", it="Italian", nl="Dutch", pt="Portuguese")
langs = sorted(k[6:] for k in me if k.startswith("mpico_"))
R, P = "pysentimiento/robertuito-irony", "cardiffnlp/twitter-roberta-base-irony"
per = "\n".join(["| language | this model: with post | reply only | F1 @ 0.5 | robertuito-irony | twitter-roberta-irony | teacher: with post |", "|---|---|---|---|---|---|---|"] +
                [f"| {LN.get(l, l)} | {f3(me['mpico_' + l]['auc'])} | {f3(me['mpico_' + l].get('auc_noctx'))} | {f3(me['mpico_' + l]['f1_05'])} | "
                 f"{f3(bl[R]['mpico_' + l]['auc'])} | {f3(bl[P]['mpico_' + l]['auc'])} | {f3(bl[T]['mpico_' + l]['auc'])} |" for l in langs])
ahead = [nm(m).split("/")[1] for m, *_ in BL if m in bl and m != T and bl[m]["_tweets_mean_auc"] > me["_tweets_mean_auc"]]
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
- ca
- sr
- hr
- bg
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
tags:
- irony
- irony-detection
- sarcasm
- sarcasm-detection
- figurative-language
- sentiment-analysis
- social-media
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Oh great, another Monday. Just what I needed."
  example_title: "Sarcastic"
- text: "Great game last night, congratulations to the whole team!"
  example_title: "Sincere"
- text: "Genial, otra vez sin luz. Me encanta este país."
  example_title: "Spanish"
- text: "Super, der Zug hat schon wieder Verspätung. Danke, Deutsche Bahn!"
  example_title: "German"
---

# Multilingual Irony & Sarcasm Detector ({size}, {PARAMS[size]})

Detects **irony and sarcasm**, meaning text whose intended meaning differs from (often is the opposite of) what it literally says:
mock praise, feigned agreement, exaggeration, deadpan. It works in English and about 45 other languages. It can read a reply
**together with the post it answers**, which is often what makes a reply ironic.
Built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}), Apache-2.0, ONNX for CPU and the browser (transformers.js) included.
{(("A larger, more accurate version is available as " if size == "small" else "A smaller, faster version is available as ") + link(OTHER) + ".") if ot else ""}

- One output, `irony` (sigmoid probability). Sincere statements, plain jokes, insults and complaints meant literally score low.
- The scores are conservative, so a threshold below 0.5 (for example 0.3) often finds more irony. Tune it on your data. {onnx}
- Useful as a signal for sentiment analysis (an ironic "great service!" is negative), content moderation and social-media research.

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}")
print(clf("Oh great, another Monday. Just what I needed."))
# with context: the post first, then the reply
print(clf({{"text": "My train was cancelled again today.", "text_pair": "Wonderful. Best rail service in the world."}}))
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await clf("Genial, otra vez sin luz. Me encanta este país."));
```

## Evaluation

Every model is scored the same way (`code/irony/eval_irony.py`): ROC AUC of its irony/sarcasm probability. No test item was used for
training, and no text from these datasets is in our training data.

- **[MultiPICo](https://huggingface.co/datasets/Multilingual-Perspectivist-NLU/MultiPICo)** (CC-BY-4.0): replies to Reddit and Twitter posts
  in 9 languages, labelled ironic or not by several annotators each (majority vote, ties dropped; up to 1,500 replies per language).
  This model gets the post as context. Reply-only scores are given for comparison, since the other models read the reply alone.
- **Tweets**: [SemEval-2018 Task 3](https://huggingface.co/datasets/cardiffnlp/tweet_eval) irony test (English), iSarcasmEval 2022 task A test
  (English), [ArSarcasm](https://huggingface.co/datasets/iabufarha/ar_sarcasm) test (Arabic) and a [Korean sarcasm](https://huggingface.co/datasets/SpellOnYou/kor_sarcasm)
  test set. Two of the English baselines were trained on SemEval-2018, so that column is in-domain for them.

{main}

- The table shows the released checkpoint. Means over {seeds}.
- On MultiPICo the existing irony and sarcasm models are close to chance (AUC about 0.5). On the tweet sets, models ahead of this one
  on the mean: {", ".join(ahead) or "none"} (teacher excluded).

MultiPICo by language (AUC):

{per}

## Training

- **Texts**: {st['by_source']['gen']:,} texts written natively in {st['n_lang']} languages by Qwen3.8-27B (Apache-2.0) for this purpose: posts on everyday
  topics with ironic, sincere and humorous-but-not-ironic replies, and standalone posts with the same mix ({st['with_context']:,} with the post
  as context). Plus {st['by_source']['civil']:,} comments from [Civil Comments](https://huggingface.co/datasets/google/civil_comments) (CC0) and their machine
  translations from [Horizon-Labs/civil-comments-multilingual](https://huggingface.co/datasets/Horizon-Labs/civil-comments-multilingual).
- **Labels**: Qwen3.8-27B answered "is this ironic or sarcastic?" for every text (with the post when there is one). The model learns that
  probability (soft labels). The checkpoint was chosen on {st['val']:,} held-out teacher-labelled texts.
- mmBERT-{size}, max length 256 tokens, (post, reply) as a text pair when a post is given. Code: `code/` in this repository.

## Limitations

- Irony is hard even for people (MultiPICo annotators often disagree), and it depends on context, tone and culture that a text alone
  may not carry. Expect errors, especially for short replies without their post.
- Labels come from an LLM teacher, not human annotators, and its errors carry over. The teacher's own scores are in the table.
- Mostly synthetic training text. Platform-specific slang and memes are under-represented.
"""
open(out, "w").write(card)
print("written", out, len(card))
