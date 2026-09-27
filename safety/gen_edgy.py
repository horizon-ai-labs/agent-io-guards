"""Synthetic benign-but-alarming prompts and harmful look-alikes (content-safety v1.2 candidate), Qwen3.8-27B via vLLM.

ITEMS: N prompts per language; each asks Qwen for 8 short user requests in one language and domain: 6 harmless requests that
use alarming words (kill, attack, shoot, bomb, drugs, poison, hack, steal...) in an ordinary sense, and 2 genuinely harmful
requests on the same topic (look-alikes, so the words alone do not decide the label). Labels come later from the teacher
(safety/label_teacher.py), not from the generator. The instruction is our own generic description; no benchmark items
(XSTest, OR-Bench test) are used.
N=... SHARD=i OUT=dir python safety/gen_edgy.py   (vLLM venv) -> $OUT/edgy_<shard>.jsonl {text, lang, intended, domain}
"""
import json, os, random, re

LANGS = ["English"] * 6 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian", "Ukrainian", "Czech",
                           "Turkish", "Arabic", "Hindi", "Chinese (Simplified)", "Japanese", "Korean", "Vietnamese", "Indonesian",
                           "Thai", "Swedish", "Hebrew"]
DOMAINS = ["programming and system administration", "cooking and food", "sports and fitness", "video games and board games",
           "medicine and pharmacy (legitimate questions)", "history and politics (factual questions)", "writing fiction and film",
           "gardening and pest control", "chemistry homework", "cars and mechanics", "music and slang", "law and crime news",
           "photography", "business and marketing", "security careers and defensive IT", "parenting and household safety"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt(lang, dom):
    return f"""We are building a content-moderation classifier and need realistic user requests to an AI assistant, written in {lang},
about {dom}.
Write 8 requests:
- 6 HARMLESS requests that contain words which sound alarming out of context (for example kill, shoot, attack, bomb, drugs,
  poison, hack, steal, weapon, execute, blow up, overdose), used in their ordinary, harmless sense for this topic.
- 2 genuinely HARMFUL requests on the same topic that a responsible assistant should refuse.
Vary length and style (some casual, some with typos); do not explain them.
Return only JSON: {{"items": [{{"text": "...", "harmful": false}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 150)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(900 + shard)
    jobs = [(l, rng.choice(DOMAINS)) for l in LANGS for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(l, d)}] for l, d in jobs],
                    SamplingParams(temperature=0.9, top_p=0.95, max_tokens=1500, seed=77 + shard), chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True)
    k = 0
    with open(f"{out}/edgy_{shard}.jsonl", "w") as f:
        for (l, d), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            for it in (j or {}).get("items", []) if isinstance(j, dict) else []:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 5 <= len(it["text"]) <= 1000:
                    f.write(json.dumps(dict(text=it["text"].strip(), lang=l, intended="harmful" if it.get("harmful") else "harmless", domain=d),
                                       ensure_ascii=False) + "\n"); k += 1
    print("items", k, "from", len(jobs), "prompts", flush=True)
