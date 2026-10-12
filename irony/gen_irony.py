"""Native texts for a multilingual irony/sarcasm classifier, Qwen3.8-27B via vLLM. Two prompt types (half each):
- thread: a short social-media post in one language on one topic, plus 10 replies: 4 ironic or sarcastic of different kinds
  (mock praise, feigned agreement, exaggeration, understatement, rhetorical questions, deadpan), 4 sincere (genuine praise or
  agreement, complaints and criticism meant literally, questions), 2 humorous but not ironic (puns, plain jokes, playful teasing);
- standalone: 10 independent posts (no context) with the same mix.
Labels come later from the teacher (irony/teacher_irony.py); the requested type is kept for statistics only. No benchmark texts.
N=... SHARD=i OUT=dir python irony/gen_irony.py -> $OUT/irony_<shard>.jsonl  (text, context, lang, topic, platform, intended)"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from gen_sent import parse_json
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

LANGS = ["English", "Spanish", "French", "German", "Italian", "Portuguese (Brazil)", "Portuguese (Portugal)", "Dutch", "Polish", "Czech",
         "Romanian", "Hungarian", "Greek", "Ukrainian", "Russian", "Swedish", "Danish", "Norwegian", "Finnish", "Turkish",
         "Arabic (Modern Standard)", "Arabic (Egyptian)", "Arabic (Gulf)", "Arabic (Levantine)", "Hebrew", "Persian", "Urdu", "Hindi",
         "Hinglish (Hindi in Latin letters mixed with English)", "Bengali", "Tamil", "Marathi", "Chinese (Simplified)", "Chinese (Traditional)",
         "Japanese", "Korean", "Vietnamese", "Thai", "Indonesian", "Malay", "Tagalog", "Swahili", "Catalan", "Serbian", "Croatian", "Bulgarian"]
TOPICS = ["public transport delays", "a football match", "a new smartphone", "the weather", "a politician's speech", "a celebrity wedding",
          "rising rent", "a restaurant visit", "a video game release", "a TV series finale", "working from home", "a tax reform", "a traffic jam",
          "a school reform", "a concert", "online shopping delivery", "a company layoff", "a vacation", "a diet trend", "an airline",
          "customer service", "a new law", "an election result", "a heatwave", "a movie premiere", "a startup launch", "a bank fee",
          "a neighbour's renovation", "a hospital waiting room", "a university exam", "a strike", "a software update", "a sports transfer"]
PLATFORMS = ["Twitter/X", "Reddit", "a news comment section", "a Facebook group", "YouTube comments", "a forum", "Instagram comments", "a group chat"]
MIX = """- 4 IRONIC or SARCASTIC texts of different kinds: mock praise, feigned agreement, exaggeration, understatement, rhetorical
  questions, deadpan statements whose real meaning is the opposite of the literal one;
- 4 SINCERE texts that mean exactly what they say: genuine praise or agreement, complaints and criticism stated literally, questions;
- 2 HUMOROUS but NOT ironic texts: puns, plain jokes or playful teasing that mean what they say."""


def prompt(kind, lang, topic, platform):
    head = f"""You are creating training data for a classifier that must recognise irony and sarcasm in {lang}. Write as native speakers
really write on {platform} (slang, abbreviations, emojis, typos where natural; do not translate from English)."""
    if kind == "thread":
        return f"""{head}
First write a short post about {topic}. Then write 10 different replies to it, in {lang}:
{MIX}
Return only JSON: {{"post": "...", "replies": [{{"text": "...", "type": "ironic|sincere|humorous"}}, ...]}}"""
    return f"""{head}
Write 10 different independent posts about {topic}, in {lang}:
{MIX}
Return only JSON: {{"replies": [{{"text": "...", "type": "ironic|sincere|humorous"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 3000)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(9900 + shard)
    jobs = [(rng.choice(["thread", "standalone"]), rng.choice(LANGS), rng.choice(TOPICS), rng.choice(PLATFORMS)) for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs], SamplingParams(temperature=0.95, top_p=0.95, max_tokens=2500, seed=900 + shard),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0; refused = 0
    clean = lambda t: t.encode("utf-8", "ignore").decode().strip()
    with open(f"{out}/irony_{shard}.jsonl", "w") as f:
        for (kind, l, tp, p), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            items = (j or {}).get("replies", []) if isinstance(j, dict) else []
            ctx = clean(j.get("post", "")) if kind == "thread" and isinstance(j, dict) and isinstance(j.get("post"), str) else ""
            refused += not items or (kind == "thread" and not ctx)
            if kind == "thread" and not ctx:
                continue
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 2 <= len(it["text"]) <= 1000:
                    f.write(json.dumps(dict(text=clean(it["text"]), context=ctx, lang=l, topic=tp, platform=p, intended=str(it.get("type", ""))), ensure_ascii=False) + "\n"); k += 1
    print("texts", k, "from", len(jobs), "prompts; empty/refused", refused, flush=True); hard_exit(0)
