"""Model card for Horizon-Labs/multilingual-sentiment-{small,base}.
python release/sentiment/make_sent_card.py SIZE RELEASE_DIR OURS_EVAL.json OTHER_SIZE_EVAL.json BASELINES.json TEACHER.json STATS.json OUT
(OURS_EVAL / OTHER_SIZE_EVAL: eval_sent.py output, first entry taken; TEACHER: merged teacher_sent.py results, prompt v2)"""
import json, os, sys

size, rd, ours_f, other_f, bl_f, te_f, st_f, out = sys.argv[1:9]
other = "base" if size == "small" else "small"
REPO, OTHER = f"Horizon-Labs/multilingual-sentiment-{size}", f"Horizon-Labs/multilingual-sentiment-{other}"
PARAMS = {"small": "141M", "base": "308M"}
first = lambda f: list(json.load(open(f)).values())[0]
me, ot, bl, te, st = first(ours_f), first(other_f), json.load(open(bl_f)), json.load(open(te_f)), json.load(open(st_f))
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: f"{x:.3f}"
FAM = [("tweets", "Tweets (8 languages, 3-class)"), ("amazon", "Amazon reviews (6 languages, 3-class)"), ("mteb", "MTEB MultilingualSentiment (29 languages, pos/neg)")]
BL = [("cardiffnlp/twitter-xlm-roberta-base-sentiment", "278M", "none given", "trained on the train split of the tweet benchmark"),
      ("cardiffnlp/twitter-xlm-roberta-base-sentiment-multilingual", "278M", "none given", "same tweet training data"),
      ("lxyuan/distilbert-base-multilingual-cased-sentiments-student", "135M", "Apache-2.0", ""),
      ("nlptown/bert-base-multilingual-uncased-sentiment", "167M", "MIT", "trained on product reviews (1-5 stars)"),
      ("tabularisai/multilingual-sentiment-analysis", "135M", "CC-BY-NC-4.0", ""),
      ("clapAI/modernBERT-base-multilingual-sentiment", "150M", "Apache-2.0", "trained on an aggregate of public sentiment sets that may include these benchmarks' train splits")]
mean = lambda r, fam: r[f"_{fam}_mean_f1"]
rows = [f"| **this model** ({PARAMS[size]}) | Apache-2.0 | " + " | ".join(f"**{f3(mean(me, f))}**" for f, _ in FAM) + " | |",
        f"| [{OTHER.split('/')[1]}](https://huggingface.co/{OTHER}) ({PARAMS[other]}) | Apache-2.0 | " + " | ".join(f3(mean(ot, f)) for f, _ in FAM) + " | |"]
for m, p, lic, note in BL:
    rows.append(f"| [{m}](https://huggingface.co/{m}) ({p}) | {lic} | " + " | ".join(f3(mean(bl[m], f)) for f, _ in FAM) + f" | {note} |")
rows.append("| Qwen3.8-27B (our teacher, zero-shot prompt) | Apache-2.0 | " + " | ".join(f3(mean(te, f)) for f, _ in FAM) + " | 27B LLM, for reference |")
main = "\n".join(["| model | licence | " + " | ".join(n for _, n in FAM) + " | note |", "|---|---|" + "---|" * len(FAM) + "---|"] + rows)
# computed comparison: which baselines are ahead of us, per family
ahead = {f: [m.split("/")[1] for m, *_ in BL if mean(bl[m], f) > mean(me, f)] for f, _ in FAM}
pl = sorted(k for k in me if not k.startswith("_"))
names = {"tweets": dict(ar="Arabic", en="English", fr="French", ge="German", hi="Hindi (romanized)", it="Italian", po="Portuguese", sp="Spanish"),
         "amazon": dict(de="German", en="English", es="Spanish", fr="French", ja="Japanese", zh="Chinese"),
         "mteb": dict(ara="Arabic", bam="Bambara", bul="Bulgarian", cmn="Chinese (cmn set)", cym="Welsh", deu="German",
                      dza="Algerian Arabic", ell="Greek", eng="English", eus="Basque", fas="Persian", fin="Finnish", heb="Hebrew",
                      hrv="Croatian", ind="Indonesian", jpn="Japanese", kor="Korean", mlt="Maltese", nor="Norwegian", pol="Polish",
                      rus="Russian", slk="Slovak", spa="Spanish", tha="Thai", tur="Turkish", uig="Uyghur", urd="Urdu",
                      vie="Vietnamese", zho="Chinese (zho set)")}
SEEDS = json.load(open(os.environ["SEEDS"])) if os.environ.get("SEEDS") else {}
C = "cardiffnlp/twitter-xlm-roberta-base-sentiment"
per = "\n".join(["| set | this model | cardiffnlp xlm-r | teacher |", "|---|---|---|---|"] +
                [f"| {k.split('_')[0]} {names.get(k.split('_')[0], {}).get(k.split('_')[1], '`' + k.split('_')[1] + '`')} | {f3(me[k]['macro_f1'])} | "
                 f"{f3(bl[C][k]['macro_f1'])} | {f3(te[k]['macro_f1'])} |" for k in pl])
weak = [k for k in pl if me[k]["macro_f1"] < 0.7]
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB) gives the same label as fp32 on "
        f"{100 * gq.get('agree', 0):.1f}% of {qs.get('n', 400)} benchmark texts.") if gq else ""

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
- ar
- he
- fa
- tr
- hi
- ur
- bn
- zh
- ja
- ko
- vi
- th
- id
- ms
- sw
- fi
- sv
- 'no'
- da
- el
- bg
- hr
- sk
- ro
- hu
- cy
- eu
- mt
- ug
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-{size}
datasets:
- HuggingFaceFW/fineweb-2
- HuggingFaceFW/fineweb
tags:
- sentiment-analysis
- sentiment
- multilingual
- text-classification
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "The battery lasts forever and the screen is gorgeous. Best purchase this year!"
  example_title: "English"
- text: "Das Paket kam zwei Wochen zu spät und war beschädigt."
  example_title: "German"
- text: "サービスは丁寧だったけど、料理は正直いまいちでした。"
  example_title: "Japanese (mixed)"
- text: "الخدمة ممتازة والموظفون لطفاء جدا"
  example_title: "Arabic"
- text: "La reunión es el martes a las diez."
  example_title: "Spanish (neutral)"
---

# Multilingual Sentiment ({size}, {PARAMS[size]}): negative / neutral / positive

A multilingual sentiment classifier built on [mmBERT-{size}](https://huggingface.co/jhu-clsp/mmBERT-{size}). It returns
**negative**, **neutral** or **positive** for reviews, social posts, comments, messages and support tickets in many
languages. It is Apache-2.0 and trained only on openly licensed text with labels from an Apache-2.0 LLM, so it can be used
commercially. It includes ONNX files for CPU and the browser (transformers.js).
{"A larger, more accurate version is available as" if size == "small" else "A smaller, faster version is available as"}
[{OTHER.split('/')[1]}](https://huggingface.co/{OTHER}). [Try it in the browser](https://huggingface.co/spaces/Horizon-Labs/multilingual-sentiment).

- Mixed or balanced opinions ("good quality but too expensive") are **neutral**. Plain facts, questions and requests are neutral too.
- The scores are probabilities, e.g. `positive - negative` gives a -1 to 1 polarity score.
- {onnx}

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}")
clf(["I love this phone, the camera is amazing!", "Le colis est arrivé cassé.", "Der Termin ist am Montag."])
# [{{'label': 'positive', ...}}, {{'label': 'negative', ...}}, {{'label': 'neutral', ...}}]
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await clf("Obrigado pela ajuda, vocês são incríveis!"));
```

## Evaluation

These are public benchmarks, used only for evaluation. We never trained on them, and any training text that also appears
in them was removed. The metric is macro-F1, averaged over the languages of each benchmark. On the two-class MTEB sets,
each model's negative-vs-positive choice is scored, ignoring neutral. Every model is run with the same script on the same
texts (`code/`), and each model's labels are mapped to negative/neutral/positive (1-2 stars = negative, 3 = neutral,
4-5 = positive; "very negative" = negative).

{main}

{("- Numbers are for the released checkpoint (seed 0). Mean of two training seeds: " + "; ".join(f"{k} {v}" for k, v in SEEDS.items()) + ".") if SEEDS else ""}
- Tweets: [cardiffnlp/tweet_sentiment_multilingual](https://huggingface.co/datasets/cardiffnlp/tweet_sentiment_multilingual)
  test (870 per language). Amazon: amazon_reviews_multi test (900 per language, balanced; 3 stars = neutral).
  MTEB: [mteb/multilingual-sentiment-classification](https://huggingface.co/datasets/mteb/multilingual-sentiment-classification)
  test (up to 600 per language; several languages are machine-translated, e.g. Welsh = translated IMDB).
- Models ahead of this one: tweets: {", ".join(ahead["tweets"]) or "none"}; Amazon: {", ".join(ahead["amazon"]) or "none"};
  MTEB: {", ".join(ahead["mteb"]) or "none"}. Models trained on a benchmark's own training split have an in-domain advantage
  on it; this model has seen no benchmark data.

Per benchmark language:

{per}

## Training

- **Text** (about {st['n_train'] // 1000}k training examples): snippets from
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) and [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
  (ODC-BY), in 67 languages, half of them from review, forum, comment and blog pages. We added synthetic reviews, posts,
  comments, messages and complaints in about 90 languages and varieties (including Hinglish and Arabic dialects), written by
  Qwen3.8-27B (Apache-2.0) in many genres, topics and lengths.
- **Labels**: soft labels (probabilities for negative, neutral and positive) from Qwen3.8-27B with a fixed zero-shot
  instruction (`code/sentiment/teacher_sent.py`). The model is trained to match these probabilities. The teacher's scores on
  the benchmarks are in the table above.
- **Model**: mmBERT-{size} with a 3-way head, 3 epochs, max length 512 tokens. The checkpoint was chosen by agreement with
  the teacher on held-out teacher-labelled data, never on the benchmarks.
- Code: `code/` in this repository.

## Limitations

- Labels come from an LLM, not from human annotators, so the model inherits the teacher's judgement. Its idea of "neutral"
  can differ from a given dataset's convention: tweet benchmarks mark many mildly opinionated posts as neutral.
- Accuracy is lower on sarcasm, on very short or context-dependent messages (tweets), and on some low-resource languages.
  {("Benchmark sets below 0.70 macro-F1: " + ", ".join(f"`{k}`" for k in weak) + ".") if weak else ""}
- Trained on texts up to 512 tokens and evaluated on texts up to 2,000 characters. The model accepts up to 8,192
  tokens, but longer documents are untested; for those, pass `truncation=True, max_length=512`, or score paragraphs
  separately.
- Sentiment is not stance, emotion or toxicity. Don't use it to make decisions about individuals.
"""
open(out, "w").write(card)
print("written", out, len(card))
