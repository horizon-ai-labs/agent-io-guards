"""Multi-sentence groundedness data (MiniCheck-style C2D / D2C), generated with Qwen3.8-27B via vLLM.

Why: on LLM-AggreFact our models trail MiniCheck in ranking quality (mean AUC .773 vs .829), most on claim verification
against retrieved documents (ClaimVerify, LFQA, Reveal). Those claims often need facts from several sentences.
  C2D: Qwen writes a claim with 2-4 atomic facts and a 6-12 sentence document that states each fact in a different
       sentence (paraphrased) among distractors, tagging which sentence supports which fact. Supported pair = full
       document; unsupported pairs = document with all sentences for one fact removed (labels exact by construction).
  D2C: a real FineWeb / FineWeb-2 passage split into numbered sentences; Qwen writes claims that each combine 2-3
       sentences and names them. Supported = full passage; unsupported = passage without one needed sentence.
Output jsonl: {text (document), text_pair (claim), label (1 supported / 0 not), source, lang}
"""
import json, os, random, re, sys

OUT = os.environ.get("OUT", "."); SHARD = int(os.environ.get("SHARD", 0))
N_C2D = int(os.environ.get("N_C2D", 100)); N_D2C = int(os.environ.get("N_D2C", 100))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")
rng = random.Random(55000 + SHARD)
LANGS = ["English"] * 16 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian", "Turkish",
                            "Arabic", "Hindi", "Chinese", "Japanese", "Korean", "Vietnamese", "Indonesian"]
TOPICS = ["a local news event", "a scientific study", "a company's quarterly results", "a sports match", "a medical trial",
          "a product's technical specifications", "a historical event", "a city travel guide", "a court case", "a team meeting",
          "a software release", "a government policy", "a biography of a researcher", "a film or album release", "a weather event",
          "a school or university", "a museum exhibition", "a transport project", "an environmental report", "a recipe and its origin"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt_c2d(topic, k, lang):
    return f"""Invent realistic details about {topic}. Write ONE claim (one or two sentences) that combines exactly {k} distinct atomic
facts (names, numbers, dates, places, causes). Then write a document of 6-12 sentences that states each fact in a DIFFERENT
sentence, paraphrased (not copied from the claim), mixed with related sentences that support none of the facts. No fact may be
stated in more than one sentence. Everything in {lang}.
Return only JSON: {{"claim": "...", "facts": ["fact 1", ...], "sentences": [{{"text": "...", "supports": [fact numbers, starting at 1]}}]}}"""


def prompt_d2c(sents, lang):
    doc = "\n".join(f"[{i + 1}] {s}" for i, s in enumerate(sents))
    return f"""Document (numbered sentences):
{doc}

Write 3 claims about this document in {lang}. Each claim must combine information from 2 or 3 DIFFERENT sentences (paraphrase,
do not copy) and be fully supported by them. For each claim list the numbers of the sentences it needs; every listed sentence
must be necessary (the claim would not be supported without it).
Return only JSON: {{"claims": [{{"claim": "...", "needs": [sentence numbers]}}]}}"""


def split_sents(t):
    parts = re.split(r"(?<=[.!?。！？])\s+", t.strip())
    return [p.strip() for p in parts if len(p.strip()) > 20]


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    passages = json.load(open(os.environ["PASSAGES"])) if os.environ.get("PASSAGES") else []
    rng.shuffle(passages)
    jobs = []
    for _ in range(N_C2D):
        t, k, l = rng.choice(TOPICS), rng.choice([2, 3, 3, 4]), rng.choice(LANGS)
        jobs.append(("C2D", dict(lang=l, k=k), prompt_c2d(t, k, l)))
    for p, l in passages:
        if len([j for j in jobs if j[0] == "D2C"]) >= N_D2C:
            break
        s = split_sents(p)
        if 5 <= len(s) <= 25:
            cl = l if rng.random() < 0.8 else "English"
            jobs.append(("D2C", dict(lang=l, sents=s), prompt_d2c(s, cl)))
    llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for _, _, p in jobs], SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2500, seed=500 + SHARD),
                    chat_template_kwargs={"enable_thinking": False})
    n = {"C2D": 0, "D2C": 0}
    with open(f"{OUT}/c2d_{SHARD}.jsonl", "w") as f:
        w = lambda **r: f.write(json.dumps(r, ensure_ascii=False) + "\n")
        for (mode, a, _), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            if not isinstance(j, dict):
                continue
            if mode == "C2D":
                sents, claim, facts = j.get("sentences"), j.get("claim"), j.get("facts")
                if not (isinstance(sents, list) and isinstance(claim, str) and isinstance(facts, list) and len(sents) >= 4):
                    continue
                try:
                    texts = [s["text"] for s in sents]; sup = [set(int(x) for x in s.get("supports", [])) for s in sents]
                except Exception:
                    continue
                K = len(facts)
                # every fact must be supported by exactly one sentence
                owners = {fi: [i for i, s in enumerate(sup) if fi in s] for fi in range(1, K + 1)}
                if K < 2 or any(len(v) != 1 for v in owners.values()):
                    continue
                w(text=" ".join(texts), text_pair=claim, label=1, source="c2d", lang=a["lang"]); n[mode] += 1
                for fi in rng.sample(range(1, K + 1), min(2, K)):
                    drop = set(owners[fi])
                    w(text=" ".join(t for i, t in enumerate(texts) if i not in drop), text_pair=claim, label=0, source="c2d", lang=a["lang"]); n[mode] += 1
            else:
                cl = j.get("claims")
                if not isinstance(cl, list):
                    continue
                S = a["sents"]
                for c in cl[:3]:
                    try:
                        needs = sorted(set(int(x) for x in c["needs"])); claim = c["claim"]
                    except Exception:
                        continue
                    if not (isinstance(claim, str) and 2 <= len(needs) <= 3 and all(1 <= x <= len(S) for x in needs)):
                        continue
                    w(text=" ".join(S), text_pair=claim, label=1, source="d2c", lang=a["lang"]); n[mode] += 1
                    drop = rng.choice(needs) - 1
                    w(text=" ".join(s for i, s in enumerate(S) if i != drop), text_pair=claim, label=0, source="d2c", lang=a["lang"]); n[mode] += 1
    print("pairs", n, "from", len(jobs), "prompts", flush=True)
