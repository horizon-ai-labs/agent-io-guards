"""Native (not translated) texts for a multilingual hate-speech classifier, Qwen3.8-27B via vLLM. Per prompt: one language, one
platform and one protected-characteristic category; 10 texts: 4 hateful toward a group of that category (explicit and implicit/coded
forms), 3 non-hateful texts that mention such groups (neutral or positive statements, news reporting, counter-speech or quoting hate
to criticise it), 3 offensive but not hate speech (insults or profanity aimed at an individual, or attacks on non-protected targets
such as politicians, sports fans or companies). Labels come later from the teacher (hate/teacher_hate.py); the requested type is kept
only for statistics. No benchmark texts or templates are shown.  N=... SHARD=i OUT=dir python hate/gen_hate.py -> $OUT/hate_<shard>.jsonl"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from gen_sent import parse_json
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

LANGS = ["English", "Spanish", "French", "German", "Italian", "Portuguese (Brazil)", "Portuguese (Portugal)", "Dutch", "Polish", "Czech",
         "Slovak", "Romanian", "Hungarian", "Greek", "Bulgarian", "Serbian", "Croatian", "Ukrainian", "Russian", "Swedish", "Danish",
         "Norwegian", "Finnish", "Turkish", "Arabic (Modern Standard)", "Arabic (Egyptian)", "Arabic (Gulf)", "Arabic (Levantine)",
         "Arabic (Maghrebi)", "Hebrew", "Persian", "Urdu", "Hindi", "Hinglish (Hindi in Latin letters mixed with English)", "Bengali",
         "Tamil", "Telugu", "Marathi", "Chinese (Simplified)", "Chinese (Traditional)", "Japanese", "Korean", "Vietnamese", "Thai",
         "Indonesian", "Malay", "Tagalog", "Swahili", "Amharic", "Hausa", "Yoruba", "Afrikaans"]
CATS = ["race or ethnicity", "religion", "nationality, national origin or migration status (immigrants, refugees)",
        "gender (women, men, transgender or non-binary people)", "sexual orientation (gay, lesbian, bisexual people)",
        "disability (physical, mental or cognitive)", "age (old people, young people)"]
PLATFORMS = ["comments under a news article", "a YouTube comment section", "an online game chat", "a football fan forum", "replies on a social network",
             "a political discussion group", "a messaging group", "a local community forum", "a meme page", "an anonymous imageboard"]


def prompt(lang, cat, platform):
    return f"""You are creating training data for a content-moderation classifier that must recognise hate speech written in {lang}.
Write 10 different realistic texts from {platform}, in {lang}, as native speakers really write them (slang, abbreviations,
dialect, emojis, typos where natural; do not translate from English). Focus on groups defined by {cat} that are actually targeted
in {lang}-speaking societies:
- 4 HATEFUL texts against such a group, of different kinds: slurs, dehumanising comparisons, harmful stereotypes, calls for exclusion,
  discrimination or violence, and implicit or coded hate (dog whistles, "jokes", "just asking questions").
- 3 NON-HATEFUL texts that mention such groups: neutral or positive statements, a news-style report, counter-speech that rejects hate,
  or a quote of a hateful remark in order to criticise it.
- 3 OFFENSIVE BUT NOT HATE SPEECH texts: insults or profanity aimed at one person without referring to the group, or harsh attacks on
  non-protected targets (politicians, a sports team's fans, a company, a government's policy).
Return only JSON: {{"texts": [{{"text": "...", "type": "hateful|non-hateful|offensive"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 3000)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(9800 + shard)
    jobs = [(rng.choice(LANGS), rng.choice(CATS), rng.choice(PLATFORMS)) for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs], SamplingParams(temperature=0.95, top_p=0.95, max_tokens=2500, seed=800 + shard),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0; refused = 0
    with open(f"{out}/hate_{shard}.jsonl", "w") as f:
        for (l, c, p), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            items = (j or {}).get("texts", []) if isinstance(j, dict) else []
            refused += not items
            for it in items:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 3 <= len(it["text"]) <= 1000:
                    t = it["text"].encode("utf-8", "ignore").decode().strip()
                    f.write(json.dumps(dict(text=t, lang=l, category=c.split(" (")[0], platform=p, intended=str(it.get("type", ""))), ensure_ascii=False) + "\n"); k += 1
    print("texts", k, "from", len(jobs), "prompts; empty/refused", refused, flush=True); hard_exit(0)
