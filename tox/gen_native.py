"""Native (not translated) toxic and non-toxic comments for the toxicity model, Qwen3.8-27B via vLLM, for training a
moderation classifier. Per prompt: 10 comments in one language and platform: 4 toxic of varied kinds (insults, obscenity,
threats, identity-based attacks, harassment) using slang and slurs native to that language, 4 non-toxic (incl. heated but civil
disagreement and blunt criticism without insults), 2 borderline. Labels come later from the teacher (tox/teacher_tox.py).
No benchmark texts are shown.  N=... SHARD=i OUT=dir python tox/gen_native.py -> $OUT/native_<shard>.jsonl"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from gen_sent import parse_json
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

# weak TextDetox languages get 3 slots, the other model languages 1
WEAK = ["Amharic", "Arabic (Modern Standard and dialects)", "Hinglish (Hindi in Latin letters mixed with English)", "Chinese (Simplified)",
        "Hebrew", "Tatar", "Ukrainian", "Japanese", "Italian", "German"]
OTHER = ["English", "French", "Spanish", "Portuguese", "Russian", "Hindi", "Dutch", "Polish", "Czech", "Romanian", "Swedish", "Danish",
         "Finnish", "Hungarian", "Greek", "Turkish", "Persian", "Urdu", "Bengali", "Korean", "Vietnamese", "Thai", "Indonesian", "Swahili",
         "Egyptian Arabic", "Chinese (Traditional)"]
LANGS = WEAK * 3 + OTHER
PLATFORMS = ["comments under a news article", "a YouTube comment section", "an online game chat", "a football fan forum", "replies on a social network",
             "a political discussion group", "product reviews", "a dating app chat", "a messaging group of classmates", "a local community forum"]


def prompt(lang, platform):
    return f"""You are creating training data for a content-moderation classifier that must recognise toxic comments written in {lang}.
Write 10 different realistic comments from {platform}, in {lang}, as native speakers really write them (slang, abbreviations,
dialect, emojis, typos where natural; do not translate from English):
- 4 TOXIC comments of different kinds: personal insults, vulgar/obscene language, a threat of violence, an attack on a group because of
  ethnicity, religion, gender or sexual orientation, harassment. Use the swear words, slurs and insults actually used in {lang}.
- 4 NON-TOXIC comments on the same topics: neutral or friendly, plus heated but civil disagreement and blunt criticism without insults.
- 2 BORDERLINE comments (sarcasm, mild rudeness, reclaimed or quoted slurs, jokes between friends).
Return only JSON: {{"comments": [{{"text": "...", "type": "toxic|non-toxic|borderline"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 3000)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(9700 + shard)
    jobs = [(rng.choice(LANGS), rng.choice(PLATFORMS)) for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs], SamplingParams(temperature=0.95, top_p=0.95, max_tokens=2500, seed=700 + shard),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0; refused = 0
    with open(f"{out}/native_{shard}.jsonl", "w") as f:
        for (l, p), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            items = (j or {}).get("comments", []) if isinstance(j, dict) else []
            refused += not items
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 3 <= len(it["text"]) <= 1000:
                    t = it["text"].encode("utf-8", "ignore").decode().strip()  # drop lone surrogates (crashed shard 1 once)
                    f.write(json.dumps(dict(text=t, lang=l, platform=p, intended=str(it.get("type", ""))), ensure_ascii=False) + "\n"); k += 1
    print("comments", k, "from", len(jobs), "prompts; empty/refused", refused, flush=True); hard_exit(0)
