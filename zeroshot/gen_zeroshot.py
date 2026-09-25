"""Synthetic zero-shot classification data (Qwen3.8-27B via vLLM) -> NLI pairs for an mmBERT zero-shot classifier.

Mode A (passages): for a FineWeb/FineWeb-2 passage, Qwen defines a classification task (topic, genre, intent,
sentiment, audience...), gives the correct label and 3-4 plausible wrong labels (incl. one near-miss).
Mode B (short texts): Qwen invents a task with 5-8 labels and writes 8 short texts (queries, reviews, headlines,
chat messages, tickets, tweets) in a given language, each with its gold label.
Output jsonl: {text, lang, task, labels: [...], gold: [...], source}
"""
import json, os, random, re

OUT = os.environ.get("OUT", "."); SHARD = int(os.environ.get("SHARD", 0))
N_A = int(os.environ.get("N_A", 100)); N_B = int(os.environ.get("N_B", 100))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")
rng = random.Random(21000 + SHARD)
LANGS = ["English"] * 12 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian", "Ukrainian",
                            "Turkish", "Arabic", "Hindi", "Chinese", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai",
                            "Czech", "Swedish", "Persian", "Hebrew", "Romanian", "Greek", "Hungarian", "Danish", "Finnish",
                            "Bengali", "Swahili", "Tagalog", "Urdu", "Malay"]
TASKS_A = ["main topic", "genre / type of text", "intended audience", "overall sentiment or tone", "purpose of the text",
           "field or industry", "is it an opinion, news, instruction, advertisement or narrative"]
TASKS_B = ["user intent for a voice assistant", "customer support ticket category", "product review sentiment (fine-grained)",
           "emotion expressed", "news headline topic", "spam vs legitimate message type", "urgency level of a request",
           "tweet stance on a given issue", "question type (factual, opinion, how-to, ...)", "complaint category for a bank",
           "e-commerce search query category", "job posting department", "medical symptom area (non-diagnostic)",
           "restaurant review aspect", "app store review issue type", "email category (billing, meeting, newsletter, ...)",
           "politeness level", "event type in a short message", "travel request type", "legal document type",
           "feedback type (bug, feature request, praise, question)", "chat message intent", "social media post topic",
           "risk or issue category", "product category of a listing", "customer satisfaction level"]
DOMAINS = ["healthcare", "banking", "video games", "travel", "education", "retail", "telecom", "automotive", "real estate",
           "human resources", "legal services", "public administration", "sports", "entertainment", "food delivery",
           "software and IT", "insurance", "energy", "agriculture", "fashion", "social media", "science", "parenting",
           "pets", "cooking", "fitness", "music", "cybersecurity", "logistics", "hospitality", "politics", "climate"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt_a(passage, task):
    return f"""Text:
<<<
{passage[:2000]}
>>>

Define a classification of this text by its {task}. Give the correct label and 4 wrong but plausible labels, one of
which is a near-miss (close to correct, but clearly wrong for this text). Labels are short (1-4 words), in English,
and could appear in a real label set. If more than one label is clearly correct, list all correct ones.
Return only JSON: {{"task": "...", "correct": ["..."], "wrong": ["...", "...", "...", "..."]}}"""


def prompt_b(task, domain, lang):
    return f"""Create data for a zero-shot text classifier.
Task: {task}, in the domain of {domain}. Invent a realistic label set of 4-10 short English labels for it (labels as
people would name them in a real product, not generic). Write a hypothesis template: an English sentence with {{}}
where a label goes, e.g. "The customer is asking about {{}}." Then write 8 diverse, realistic short texts (1-3
sentences each) in {lang}, the kind that appear in practice (varied length and style, some informal or with typos).
Give each text's correct label; use as many different labels as possible.
Return only JSON: {{"task": "...", "labels": ["..."], "template": "...", "items": [{{"text": "...", "label": "..."}}]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    passages = json.load(open(os.environ["PASSAGES"])) if os.environ.get("PASSAGES") else []
    rng.shuffle(passages)
    jobs = []
    for p, l in passages[:N_A]:
        t = rng.choice(TASKS_A)
        jobs.append(("A", dict(text=p, lang=l, task=t), prompt_a(p, t)))
    for _ in range(N_B):
        t, d, l = rng.choice(TASKS_B), rng.choice(DOMAINS), rng.choice(LANGS)
        jobs.append(("B", dict(task=t, lang=l), prompt_b(t, d, l)))
    llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for _, _, p in jobs], SamplingParams(temperature=0.9, top_p=0.95, max_tokens=2000, seed=900 + SHARD),
                    chat_template_kwargs={"enable_thinking": False})
    n = 0
    with open(f"{OUT}/zs_{SHARD}.jsonl", "w") as f:
        for (mode, a, _), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            if not isinstance(j, dict):
                continue
            if mode == "A" and isinstance(j.get("correct"), list) and isinstance(j.get("wrong"), list):
                corr = [c for c in j["correct"] if isinstance(c, str)]
                wrong = [w for w in j["wrong"] if isinstance(w, str) and w not in corr]
                if corr and wrong:
                    f.write(json.dumps(dict(text=a["text"], lang=a["lang"], task=a["task"], labels=corr + wrong, gold=corr, source="passage"), ensure_ascii=False) + "\n"); n += 1
            elif mode == "B" and isinstance(j.get("labels"), list) and isinstance(j.get("items"), list):
                labels = [x for x in j["labels"] if isinstance(x, str)]
                for it in j["items"]:
                    if isinstance(it, dict) and isinstance(it.get("text"), str) and it.get("label") in labels:
                        f.write(json.dumps(dict(text=it["text"], lang=a["lang"], task=j.get("task", ""), labels=labels, gold=[it["label"]], template=j.get("template") if isinstance(j.get("template"), str) else "", source="short"), ensure_ascii=False) + "\n"); n += 1
    print(f"rows {n} from {len(jobs)} prompts", flush=True)
