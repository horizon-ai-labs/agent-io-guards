"""Zero-shot data v2: generic label taxonomies (Qwen3.8-27B via vLLM).

v1 used invented, specific label sets; the model then linked surface wording better than abstract categories
("wifi kept dropping" -> "internet" low). v2 adds data where the labels are broad, reusable categories:
  Mode C (passages): a FineWeb / FineWeb-2 passage is labelled against 2 generic taxonomies (topic, text type,
          sentiment, audience, ...). Qwen picks the correct label(s) from the full list.
  Mode D (short texts): Qwen writes 10 short texts in a language for a generic taxonomy (emotion, sentiment, topic of
          a question, news section, customer-message topic, ...), each with its label; plus abstract "aspects"
          (multi-label: which aspects does a review mention, using category words that do not appear in the text).
The texts are FineWeb or newly written; no benchmark data. Output jsonl like gen_zeroshot.py:
{text, lang, task, labels, gold, source, template}
prep:  python zeroshot/gen_zeroshot_v2.py prep OUT.json   (passages; shared python with datasets)
"""
import json, os, random, re, sys

OUT = os.environ.get("OUT", "."); SHARD = int(os.environ.get("SHARD", 0))
N_C = int(os.environ.get("N_C", 100)); N_D = int(os.environ.get("N_D", 100))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")
rng = random.Random(33000 + SHARD)
LANGS = ["English"] * 14 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian", "Ukrainian",
                            "Turkish", "Arabic", "Hindi", "Chinese", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai",
                            "Czech", "Swedish", "Persian", "Hebrew", "Romanian", "Greek", "Hungarian", "Danish", "Finnish",
                            "Bengali", "Swahili", "Tagalog", "Urdu", "Malay"]
# generic taxonomies: (name, template, labels)
TAX_C = [
    ("topic", "This text is about {}.", ["politics", "business and finance", "sports", "science", "technology", "entertainment",
     "health and medicine", "travel and tourism", "education", "crime and law", "environment and nature", "food and drink",
     "religion and spirituality", "history", "arts and culture", "home and garden", "cars and transportation", "fashion and beauty",
     "family and parenting", "jobs and careers", "real estate", "gaming", "personal finance", "pets and animals"]),
    ("text type", "This text is {}.", ["a news report", "an opinion piece", "a product description", "an advertisement",
     "instructions or a how-to guide", "a personal story", "an academic or scientific text", "a legal or official document",
     "a forum or social media post", "a review", "a biography", "a recipe", "an FAQ or help page", "a press release",
     "an event announcement", "fiction"]),
    ("sentiment", "The sentiment of this text is {}.", ["positive", "negative", "neutral", "mixed"]),
    ("audience", "This text is written for {}.", ["children", "the general public", "experts or professionals", "students",
     "customers", "investors", "patients", "tourists"]),
    ("purpose", "The purpose of this text is to {}.", ["inform", "persuade", "sell something", "entertain", "instruct",
     "ask for help", "complain", "announce an event"]),
    ("news section", "This article belongs to the {} section.", ["world news", "national news", "business", "technology",
     "science", "sports", "arts and culture", "opinion", "health", "lifestyle", "local news", "weather"]),
]
TAX_D = [
    ("emotion", "social media posts or diary lines", "This text expresses {}.", ["joy", "sadness", "anger", "fear", "surprise",
     "disgust", "love", "gratitude", "disappointment", "pride", "anxiety", "relief", "neutral"]),
    ("sentiment", "product or service reviews", "The sentiment of this review is {}.", ["positive", "negative", "neutral", "mixed"]),
    ("question topic", "questions people ask on a Q&A site", "This question is about {}.", ["society and culture",
     "science and mathematics", "health", "education", "computers and internet", "sports", "business and finance",
     "entertainment and music", "family and relationships", "politics and government", "travel", "food and cooking",
     "cars", "pets", "beauty and style", "games and recreation"]),
    ("news headline section", "news headlines with a one-sentence lead", "This news is about {}.", ["world news", "business",
     "sports", "science and technology", "politics", "entertainment", "health", "crime", "environment", "education"]),
    ("customer message topic", "customer messages to a company (chat, email)", "This customer message is about {}.",
     ["billing or payment", "delivery or shipping", "refund or return", "technical problem", "account or login",
     "product information", "complaint about staff", "cancellation", "praise or thanks", "order change"]),
    ("urgency", "messages or requests", "The urgency of this message is {}.", ["low", "medium", "high", "critical"]),
    ("formality", "messages", "The tone of this message is {}.", ["formal", "informal", "friendly", "rude", "neutral"]),
    ("spam", "messages people receive (SMS, email, DM)", "This message is {}.", ["spam", "a phishing attempt",
     "a personal message", "a work message", "a newsletter", "a notification"]),
]
ASPECT_DOMAINS = [("hotel", ["cleanliness", "staff", "internet", "food", "price", "location", "noise", "room size", "comfort"]),
                  ("restaurant", ["food quality", "service", "price", "ambience", "waiting time", "portion size", "cleanliness", "drinks"]),
                  ("smartphone", ["battery", "camera", "screen", "performance", "price", "design", "sound", "software"]),
                  ("online shop", ["delivery", "price", "product quality", "customer service", "packaging", "website", "returns"]),
                  ("airline", ["punctuality", "staff", "seat comfort", "food", "luggage", "price", "entertainment", "booking"]),
                  ("software", ["usability", "performance", "bugs", "features", "price", "documentation", "support", "design"]),
                  ("course or class", ["teacher", "content", "difficulty", "price", "materials", "schedule", "exams", "workload"]),
                  ("car", ["fuel consumption", "comfort", "reliability", "price", "design", "safety", "handling", "space"])]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt_c(passage, taxes):
    q = "\n".join(f'- "{n}": choose from {json.dumps(ls)}' for n, _, ls in taxes)
    keys = ", ".join(f'"{n}": ["..."]' for n, _, _ in taxes)
    return f"""Text:
<<<
{passage[:2000]}
>>>

Classify this text for each category scheme below. For each scheme give the label that fits best; add a second
label only if it clearly fits too. Use the labels exactly as written.
{q}
Return only JSON: {{{keys}}}"""


def prompt_d(name, kind, labels, lang):
    return f"""Write 10 realistic, varied {kind} in {lang} for a {name} classifier. Each must clearly belong to exactly one
of these labels: {json.dumps(labels)}. Use as many different labels as possible. Do NOT use the label word itself or
an obvious translation of it in the text: express it through content, as real people do. Vary length (a few words to
3 sentences), style and register; some informal, some with typos.
Return only JSON: {{"items": [{{"text": "...", "label": "..."}}]}}"""


def prompt_aspect(domain, aspects, lang):
    return f"""Write 8 realistic reviews of a {domain} in {lang}, 1-3 sentences each. Each review mentions 1-3 of these
aspects: {json.dumps(aspects)}. Refer to the aspects concretely, WITHOUT using the aspect words themselves (e.g. for
"internet" write about the wifi dropping; for "price" write "way too expensive"). List the aspects each review mentions.
Return only JSON: {{"items": [{{"text": "...", "aspects": ["..."]}}]}}"""


def load_passages(n):
    from datasets import load_dataset
    codes = {"en": None, "deu_Latn": "German", "fra_Latn": "French", "spa_Latn": "Spanish", "por_Latn": "Portuguese",
             "ita_Latn": "Italian", "nld_Latn": "Dutch", "pol_Latn": "Polish", "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian",
             "tur_Latn": "Turkish", "arb_Arab": "Arabic", "hin_Deva": "Hindi", "cmn_Hani": "Chinese", "jpn_Jpan": "Japanese",
             "kor_Hang": "Korean", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian", "tha_Thai": "Thai", "ces_Latn": "Czech",
             "swe_Latn": "Swedish", "fas_Arab": "Persian", "heb_Hebr": "Hebrew", "ell_Grek": "Greek", "ben_Beng": "Bengali",
             "swh_Latn": "Swahili"}
    per = max(1, n // (len(codes) + 7))
    out = []
    for code, name in codes.items():
        if code == "en":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True); name, k = "English", per * 8
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-2", code, split="train", streaming=True); k = per
        skip, got = 10000 + SHARD * k * 3, 0   # past the passages used for groundedness / zero-shot v1 (< 5k rows per language)
        for i, r in enumerate(ds):
            if i < skip:
                continue
            t = r["text"].strip()
            if not 300 <= len(t) <= 6000:
                continue
            L = rng.choice([400, 800, 1500, 2500])
            if len(t) > L:
                s = t.find("\n", rng.randint(0, len(t) - L))
                t = t[s + 1: s + 1 + L] if s >= 0 else t[:L]
            out.append((t, name)); got += 1
            if got >= k:
                break
        print("passages", code, got, flush=True)
    rng.shuffle(out)
    return out[:n]


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "prep":
    json.dump(load_passages(N_C), open(sys.argv[2], "w"), ensure_ascii=False)
    os._exit(0)   # streaming datasets threads crash the interpreter at shutdown (PyGILState_Release)
elif __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    passages = json.load(open(os.environ["PASSAGES"])) if os.environ.get("PASSAGES") else []
    jobs = []
    for p, l in passages[:N_C]:
        taxes = rng.sample(TAX_C[:2], 1) + rng.sample(TAX_C[2:], 1) if rng.random() < 0.7 else rng.sample(TAX_C, 2)
        jobs.append(("C", dict(text=p, lang=l, taxes=taxes), prompt_c(p, taxes)))
    for _ in range(N_D):
        l = rng.choice(LANGS)
        if rng.random() < 0.2:
            dom, asp = rng.choice(ASPECT_DOMAINS)
            jobs.append(("E", dict(lang=l, domain=dom, aspects=asp), prompt_aspect(dom, asp, l)))
        else:
            name, kind, tpl, labels = rng.choice(TAX_D)
            jobs.append(("D", dict(lang=l, name=name, tpl=tpl, labels=labels), prompt_d(name, kind, labels, l)))
    llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for _, _, p in jobs], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2500, seed=700 + SHARD),
                    chat_template_kwargs={"enable_thinking": False})
    n = 0
    with open(f"{OUT}/zs2_{SHARD}.jsonl", "w") as f:
        w = lambda **r: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for (mode, a, _), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            if not isinstance(j, dict):
                continue
            if mode == "C":
                for name, tpl, labels in a["taxes"]:
                    g = j.get(name)
                    g = [x for x in (g if isinstance(g, list) else [g]) if isinstance(x, str) and x in labels][:2]
                    if g:
                        w(text=a["text"], lang=a["lang"], task=name, labels=labels, gold=g, template=tpl, source="tax_passage"); n += 1
            elif isinstance(j.get("items"), list):
                for it in j["items"]:
                    if not (isinstance(it, dict) and isinstance(it.get("text"), str) and len(it["text"]) > 5):
                        continue
                    if mode == "D" and it.get("label") in a["labels"]:
                        w(text=it["text"], lang=a["lang"], task=a["name"], labels=a["labels"], gold=[it["label"]], template=a["tpl"], source="tax_short"); n += 1
                    elif mode == "E" and isinstance(it.get("aspects"), list):
                        g = [x for x in it["aspects"] if x in a["aspects"]]
                        if g:
                            w(text=it["text"], lang=a["lang"], task="aspects", labels=a["aspects"], gold=g,
                              template=rng.choice(["The review mentions {}.", "This review talks about {}.", "This text is about {}."]), source="aspect"); n += 1
    print(f"rows {n} from {len(jobs)} prompts", flush=True)
