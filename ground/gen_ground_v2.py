"""Groundedness data round 2: genre-targeted documents + claim-level labels (Qwen3.8-27B via vLLM).

v1 lost to MiniCheck on LLM-AggreFact, whose documents are news, meeting/dialogue transcripts, long-form QA
evidence and expert answers, with 1-3 sentence claims. Here Qwen writes a document of such a genre (in one of 30
languages) and a set of claims about it: supported ones (including multi-hop combinations and paraphrases) and
unsupported ones (contradiction, entity/number swap, unverifiable-but-plausible addition, wrong attribution in
dialogues, over-generalization). Output jsonl: {doc, lang, genre, claims: [{text, label, type}]}
"""
import json, os, random, re, sys, time

OUT = os.environ.get("OUT", "."); SHARD = int(os.environ.get("SHARD", 0)); N = int(os.environ.get("N", 200))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")
rng = random.Random(12000 + SHARD)
LANGS = ["English"] * 12 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian", "Ukrainian",
                            "Turkish", "Arabic", "Hindi", "Chinese", "Japanese", "Korean", "Vietnamese", "Indonesian", "Thai",
                            "Czech", "Swedish", "Persian", "Hebrew", "Romanian", "Greek", "Hungarian", "Danish", "Finnish"]
GENRES = ["news article", "multi-speaker meeting transcript", "customer support chat transcript", "podcast / interview transcript",
          "three retrieved web snippets answering a user question", "scientific paper abstract and results paragraph",
          "product page with several user reviews", "contract / terms clause excerpt", "medical guideline excerpt",
          "company earnings report excerpt", "Wikipedia-style biography", "technical documentation page",
          "court decision summary", "email thread between colleagues", "recipe with nutrition facts table",
          "sports match report with statistics", "government policy announcement", "travel guide section"]
TOPICS = ["technology", "health", "finance", "politics", "science", "sports", "culture", "education", "environment",
          "law", "business", "history", "travel", "food", "energy", "transport", "space", "agriculture"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt(genre, topic, lang):
    return f"""You create data for a model that checks whether a claim is fully supported by a document.

1. Write a realistic {genre} about {topic}, in {lang}, 250-600 words, with concrete names, numbers, dates and
   (for transcripts) several named speakers.
2. Write 10 claims about it in {lang}, each 1-3 sentences, the way an AI summary or answer would state them:
   - 5 SUPPORTED claims: fully entailed by the document. Include paraphrases, claims that combine two facts from
     different parts of the document, and claims that correctly attribute statements to speakers.
   - 5 UNSUPPORTED claims, each a different kind: contradiction; swapped entity or number; plausible detail that the
     document never mentions; statement attributed to the wrong speaker or source; over-generalization or wrong
     causal link. Keep them subtle and fluent; most of each claim should be correct.
Return only JSON: {{"document": "...", "claims": [{{"text": "...", "label": "supported" or "unsupported", "type": "..."}}]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    jobs = [(rng.choice(GENRES), rng.choice(TOPICS), rng.choice(LANGS)) for _ in range(N)]
    llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=3500, seed=700 + SHARD)
    t = time.time()
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs], sp, chat_template_kwargs={"enable_thinking": False})
    print(f"generated {len(outs)} in {time.time()-t:.0f}s", flush=True)
    ok = 0
    with open(f"{OUT}/ground2_{SHARD}.jsonl", "w") as f:
        for (genre, topic, lang), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            if not (isinstance(j, dict) and isinstance(j.get("document"), str) and isinstance(j.get("claims"), list)):
                continue
            claims = [dict(text=c["text"], label=1 if str(c.get("label", "")).lower().startswith("supported") else 0, type=c.get("type", ""))
                      for c in j["claims"] if isinstance(c, dict) and isinstance(c.get("text"), str) and len(c["text"]) > 15]
            if sum(c["label"] for c in claims) == 0 or all(c["label"] for c in claims):
                continue
            f.write(json.dumps(dict(doc=j["document"], lang=lang, genre=genre, claims=claims), ensure_ascii=False) + "\n")
            ok += 1
    print(f"ok {ok}/{len(jobs)}", flush=True)
