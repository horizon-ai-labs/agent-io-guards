"""Synthetic newswire-style snippets for NER training (Qwen3.8-27B via vLLM; Apache-2.0 outputs). Labels come later from the
teacher (ner/teacher_ner.py), not from the generator.   N=... SHARD=i OUT=dir python ner/gen_news.py -> $OUT/news_<shard>.jsonl
Genres cover what web prose lacks: wire leads with datelines, headlines (some in capitals), sports results and tables,
captions, encyclopedic first sentences, market and election briefs. The instruction asks for many different, realistic
names of people, organisations, places, events and products.
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
LANGS = (["English"] * 4 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Russian", "Polish", "Chinese (Simplified)",
         "Japanese", "Arabic", "Turkish"] * 2 +
         ["Czech", "Ukrainian", "Romanian", "Swedish", "Danish", "Norwegian", "Finnish", "Hungarian", "Greek", "Bulgarian", "Croatian",
          "Serbian", "Hebrew", "Persian", "Hindi", "Bengali", "Korean", "Vietnamese", "Thai", "Indonesian", "Malay", "Swahili", "Tagalog",
          "Catalan", "Slovak", "Lithuanian", "Estonian", "Latvian", "Slovenian", "Urdu", "Tamil", "Telugu", "Marathi"])
GENRES = ["the first sentence of a wire news story starting with a city dateline (e.g. 'BERLIN 2024-03-11 -' or 'PARIS (agency) -')",
          "a news headline written entirely in CAPITAL LETTERS", "a normal news headline", "a football or basketball result line "
          "(teams, score, scorers with minutes)", "a row of a sports league table or a race result (rank, name, team or country, points or time)",
          "a photo caption naming people and the place", "the first sentence of an encyclopedia article about a person, company, city or work",
          "a short stock-market or business brief naming companies and executives", "a short election or politics brief naming parties and politicians",
          "a short culture or film news item naming artists, works and festivals"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    from common import hard_exit
    n, shard, out = int(os.environ.get("N", 120)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(4100 + shard)
    jobs = [(l, rng.choice(GENRES)) for l in LANGS for _ in range(n)]
    prompts = [f"Write 10 different realistic examples of {g}, in {l}. Use many different, plausible names of people, organisations, "
               f"places, events and products (local and international, not only famous ones). Write as a native {l} news writer; do not "
               "translate from English. Return only JSON: {\"items\": [\"...\", ...]}" for l, g in jobs]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SamplingParams(temperature=0.95, top_p=0.95, max_tokens=2000, seed=71 + shard),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    os.makedirs(out, exist_ok=True); k = 0
    with open(f"{out}/news_{shard}.jsonl", "w") as f:
        for (l, g), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            for it in (j or {}).get("items", []) if isinstance(j, dict) else []:
                if isinstance(it, str) and 10 <= len(it) <= 600:
                    f.write(json.dumps(dict(text=it.strip(), lang=l, genre=g[:40]), ensure_ascii=False) + "\n"); k += 1
    print("items", k, "from", len(jobs), "prompts", flush=True)
    hard_exit(0)
