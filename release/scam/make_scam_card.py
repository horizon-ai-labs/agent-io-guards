"""Model card for Horizon-Labs/phishing-scam-detector-base.
python release/scam/make_scam_card.py RELEASE_DIR OURS_EVAL.json SEEDS.json BASELINES.json STATS.json OUT"""
import json, os, sys
import numpy as np

rd, ours_f, seeds_f, bl_f, st_f, out = sys.argv[1:7]
REPO = "Horizon-Labs/phishing-scam-detector-base"
me = list(json.load(open(ours_f)).values())[0]
seeds, bl, st = json.load(open(seeds_f)), json.load(open(bl_f)), json.load(open(st_f))
qs = json.load(open(f"{rd}/onnx_sweep.json")) if os.path.exists(f"{rd}/onnx_sweep.json") else {}
f3 = lambda x: "—" if x is None else f"{x:.3f}"
SM = ["smish_bn", "smish_pt", "smish_ko"]
fpr = lambda r: float(np.mean([r[s]["fpr"] for s in SM]))
BL = [("cybersectony/phishing-email-detection-distilbert_v2.4.1", "67M", "Apache-2.0", "English emails and URLs"),
      ("ptouch/phishing-distilbert-cyber207", "67M", "none given", "English"),
      ("ealvaradob/bert-finetuned-phishing", "110M", "Apache-2.0", "English emails, SMS, URLs, websites"),
      ("mshenoda/roberta-spam", "125M", "MIT", "English spam messages"),
      ("mrm8488/bert-tiny-finetuned-sms-spam-detection", "4M", "none given", "trained on the UCI SMS set (in-domain)"),
      ("mrm8488/bert-tiny-finetuned-enron-spam-detection", "4M", "Apache-2.0", "trained on Enron spam (in-domain)"),
      ("Qwen/Qwen3.8-27B (teacher, prompted)", "27B", "Apache-2.0", "our teacher, for reference")]
link = lambda m: m if " " in m else f"[{m}](https://huggingface.co/{m})"
row = lambda name, p, lic, r, note: (f"| {name} ({p}) | {lic} | {f3(r['_smish_auc'])} | {f3(r['_smish_f1'])} | {f3(fpr(r))} | "
                                      f"{f3(r['_email_auc'])} | {f3(r['_sms_en_auc'])} | {f3(r['_sms_multi_auc'])} | {note} |")
rows = [row("**this model**", "307M", "Apache-2.0", me, "multilingual")] + [row(link(m), p, lic, bl[m], note) for m, p, lic, note in BL if m in bl]
main = "\n".join(["| model | licence | real smishing, 3 languages: AUC | F1 @ 0.5 | false-positive rate @ 0.5 | phishing & spam emails (English): AUC "
                  "| SMS spam (English): AUC | SMS spam, 21 languages (machine-translated): AUC | note |",
                  "|---|---|---|---|---|---|---|---|---|"] + rows)
NAME = {"smish_bn": "Bengali / Banglish / English SMS (smishing + promotions)", "smish_pt": "Mozambican Portuguese SMS and Facebook (smishing)",
        "smish_ko": "Korean SMS (smishing)", "phish_email": "phishing vs safe emails (English)", "enron_spam": "Enron spam vs ham (English)",
        "sms_en": "UCI SMS spam (English)"}
per = "\n".join(["| set | AUC | F1 @ 0.5 | false-positive rate | recall |", "|---|---|---|---|---|"] +
                [f"| {NAME[s]} | {f3(me[s]['auc'])} | {f3(me[s]['f1'])} | {f3(me[s]['fpr'])} | {f3(me[s]['recall'])} |" for s in SM + ["phish_email", "enron_spam", "sms_en"]])
langs = sorted(k[4:] for k in me if k.startswith("sms_") and k != "sms_en")
multi = ", ".join(f"{l} {me['sms_' + l]['auc']:.3f}" for l in langs)
sd = [v for k, v in seeds.items() if not k.startswith("small")]
gq = qs.get("gather_only", {})
onnx = (f"`onnx/model_quantized.onnx` (int8 embeddings, {gq.get('mb', 0):.0f} MB) gives the same top label as fp32 for "
        f"{100 * gq.get('agree', 0):.1f}% of {qs.get('n', 0)} benchmark texts.") if gq else ""

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
- tr
- ar
- fa
- he
- hi
- bn
- ur
- mr
- pa
- zh
- ja
- ko
- vi
- th
- id
- ms
- jv
- sw
- sv
- 'no'
- da
- fi
- cs
- ro
- hu
- el
library_name: transformers
pipeline_tag: text-classification
base_model: jhu-clsp/mmBERT-base
tags:
- phishing
- phishing-detection
- scam-detection
- smishing
- spam-detection
- fraud-detection
- email-security
- multilingual
- modernbert
- mmbert
- onnx
- transformers.js
widget:
- text: "Your parcel could not be delivered. Pay the 1.99 EUR customs fee within 24h: dhl-redelivery.example.com"
  example_title: "Parcel scam"
- text: "Hi, running 10 min late, see you at the cafe"
  example_title: "Normal message"
- text: "Ihr Konto wurde gesperrt. Bestätigen Sie Ihre Daten innerhalb von 24 Stunden: sparkasse-sicher.example.de"
  example_title: "German phishing"
- text: "MEGA SALE 70% off all shoes this weekend only! Shop now"
  example_title: "Marketing spam"
---

# Phishing & Scam Detector (base, 307M): multilingual

Flags **phishing, scams and spam** in emails, SMS, chat and social-media messages in many languages. Three labels:

- **`fraud`**: phishing (fake bank, delivery, account or tax messages with links), scams (prizes, investment and crypto schemes,
  fake jobs, "hi mum, new number", romance, advance fee), requests for passwords, one-time codes or payment details, malware lures;
- **`spam`**: unwanted advertising and bulk marketing that is not trying to steal anything;
- **`legitimate`**: everything else, including genuine notifications that look similar (real one-time codes, delivery updates,
  bank alerts).

Built on [mmBERT-base](https://huggingface.co/jhu-clsp/mmBERT-base), Apache-2.0, ONNX files for CPU and the browser
(transformers.js) included. Useful for message filtering and moderation, and for AI agents that read email or chats (see also
our [prompt-injection guard](https://huggingface.co/Horizon-Labs/prompt-injection-guard-base) for attacks on the agent itself).

- Score: `P(fraud) + P(spam)` for "unwanted", or `P(fraud)` alone if marketing is acceptable. The default threshold is 0.5;
  raise it to cut false positives (see the false-positive rates below).
- {onnx}

## Usage

```python
from transformers import pipeline

clf = pipeline("text-classification", model="{REPO}", top_k=None)
print(clf("Your parcel could not be delivered. Pay the 1.99 EUR customs fee within 24h: dhl-redelivery.example.com"))
# [[{{'label': 'fraud', 'score': ...}}, {{'label': 'legitimate', 'score': ...}}, {{'label': 'spam', 'score': ...}}]]
```

transformers.js:

```js
import {{ pipeline }} from "@huggingface/transformers";
const clf = await pipeline("text-classification", "{REPO}", {{ dtype: "q8" }});
console.log(await clf("Ihr Konto wurde gesperrt. Bestätigen Sie Ihre Daten: sparkasse-sicher.example.de", {{ top_k: null }}));
```

## Evaluation

Public spam / phishing / smishing test sets, used only for evaluation. Every model is scored the same way (`code/`): AUC of its
"unwanted" probability (for this model `P(fraud) + P(spam)`), and F1 and false-positive rate at 0.5.

- **Real smishing in other languages** (the main test): Bengali / Banglish / code-mixed SMS
  ([shariul-islam/bengali-sms-smishing-dataset](https://huggingface.co/datasets/shariul-islam/bengali-sms-smishing-dataset), test split; smishing and promotions count as unwanted),
  Mozambican Portuguese SMS and Facebook messages ([MOZNLP/MOZ-Smishing](https://huggingface.co/datasets/MOZNLP/MOZ-Smishing)),
  Korean SMS ([jmjmjm3/kor-smishing-message](https://huggingface.co/datasets/jmjmjm3/kor-smishing-message)), up to 1,000 each.
- **English emails**: [zefang-liu/phishing-email-dataset](https://huggingface.co/datasets/zefang-liu/phishing-email-dataset) and
  [SetFit/enron_spam](https://huggingface.co/datasets/SetFit/enron_spam) test; **English SMS**: the UCI SMS Spam Collection; and its
  machine translation into 21 languages ([dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset](https://huggingface.co/datasets/dbarbedillo/SMS_Spam_Multilingual_Collection_Dataset)).
  These classic sets are in or near the training data of several English baselines (marked in the note column), which explains
  their near-perfect English scores.

{main}

- The table shows the released checkpoint. Two training seeds: real-smishing AUC {f3(sd[0]['_smish_auc'])} / {f3(sd[1]['_smish_auc'])},
  false-positive rate {f3(fpr(sd[0]))} / {f3(fpr(sd[1]))}.
- On real smishing in Bengali, Portuguese and Korean this model ranks messages much better (AUC) and flags far fewer normal
  messages than the English models (their false-positive rates there: 0.33-0.96). Its F1 at 0.5 is lower than roberta-spam's
  because it is conservative at 0.5: on the Portuguese set it catches only {me['smish_pt']['recall']:.0%} of smishing messages at
  0.5 (Bengali {me['smish_bn']['recall']:.0%}, Korean {me['smish_ko']['recall']:.0%}). Lower the threshold if missed scams cost more than false alarms.
- On the classic English sets it is slightly behind models trained on them.
- Machine-translated SMS by language (AUC): {multi}.
- The `spam` / `fraud` split comes from our teacher; the public sets only distinguish wanted from unwanted, so only that is measured.

Per set:

{per}

## Training

- **Messages**: {st['n_gen']:,} synthetic SMS, emails, chat, social-media and push messages in about 70 languages written by
  Qwen3.8-27B (Apache-2.0): per prompt 4 legitimate (half of them genuine notifications that resemble phishing), 2 spam and 4 fraud
  of one of 20 scam types (parcel fees, bank verification, tax refunds, prizes, fake jobs, crypto, romance, "new number", CEO fraud,
  fake login alerts, OTP theft, toll fines, ...). Plus {st['n_fw']:,} ordinary web snippets in 92 languages from
  [FineWeb-2](https://huggingface.co/datasets/HuggingFaceFW/fineweb-2) / [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)
  (ODC-BY), so normal text stays normal. No test-set messages were used.
- **Labels**: Qwen3.8-27B (prompted) scored every text as legitimate / spam / fraud; the model learns the teacher's probabilities
  (soft labels). Checkpoint chosen on held-out teacher-labelled data.
- mmBERT-base, max length 384 tokens, learning rate 3e-5. Code: `code/` in this repository.

## Limitations

- Not a complete security product: it sees only the text, not sender reputation, links, attachments or headers. Use it as one
  signal among others, and review decisions that matter.
- False positives remain (about {fpr(me):.0%} of normal messages on the smishing test sets at 0.5), especially for genuine
  notifications with links and for promotions; tune the threshold on your own traffic.
- Trained on synthetic messages: new scam scripts, very long emails (truncated at 384 tokens) and rare languages are weaker. The
  Portuguese smishing set is the hardest here (AUC {me['smish_pt']['auc']:.3f}; at 0.5 it misses most of those messages).
"""
open(out, "w").write(card)
print("written", out, len(card))
