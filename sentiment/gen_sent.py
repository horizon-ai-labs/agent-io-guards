"""Synthetic multilingual opinionated texts for sentiment training, Qwen3.8-27B via vLLM (Apache-2.0 outputs).
Each prompt asks for 10 texts in one language, genre and topic: 3 clearly positive, 3 clearly negative, 2 neutral, 2 hard
(mixed, sarcastic, subtle or negated). Labels come later from the teacher (sentiment/teacher_sent.py), not the generator.
No benchmark items are shown to the generator.
N=... SHARD=i OUT=dir python sentiment/gen_sent.py   (vLLM venv) -> $OUT/gen_<shard>.jsonl {text, lang, genre, topic, intended}
LONG=1: 4 long texts per prompt (120-350 words: reviews, posts, letters), 1 positive, 1 negative, 1 neutral, 1 hard - the
v0 student trailed the teacher most on long reviews (e.g. mteb_cym, translated IMDB) because most texts were short.
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import hard_exit

LANGS = (["English"] * 6 + ["German", "French", "Spanish", "Portuguese (Brazilian)", "Italian", "Arabic (Modern Standard)",
         "Egyptian Arabic", "Hindi", "Hinglish (Hindi in Latin letters mixed with English)", "Chinese (Simplified)", "Chinese (Traditional)",
         "Japanese", "Korean", "Russian"] * 2 +
         ["Dutch", "Polish", "Ukrainian", "Czech", "Slovak", "Bulgarian", "Croatian", "Serbian", "Slovenian", "Romanian", "Hungarian",
          "Greek", "Turkish", "Hebrew", "Persian", "Urdu", "Bengali", "Tamil", "Telugu", "Marathi", "Vietnamese", "Thai",
          "Indonesian", "Malay", "Filipino (Tagalog)", "Swahili", "Finnish", "Swedish", "Norwegian", "Danish", "Welsh", "Basque",
          "Catalan", "Maltese", "Lithuanian", "Latvian", "Estonian", "Uyghur", "Algerian Arabic (Darja)", "Moroccan Arabic (Darija)",
          "European Portuguese", "Latin American Spanish", "Amharic", "Hausa", "Yoruba", "Nepali", "Punjabi", "Gujarati", "Kazakh",
          "Azerbaijani", "Georgian", "Armenian", "Albanian", "Icelandic", "Irish"])
GENRES = ["product review on a shopping site", "restaurant or cafe review", "hotel or travel review", "movie, series or book review",
          "app store review", "short social media post", "reply in a social media thread", "comment under a news article",
          "customer support message or complaint", "chat message to a friend", "forum post", "short email to a company",
          "answer in a customer satisfaction survey", "comment about a sports match", "opinion about a politician or policy",
          "comment on a video or song", "message to a landlord, school or employer", "post about a personal life event"]
TOPICS = ["electronics", "clothing and shoes", "food and groceries", "a local business", "public transport", "a bank or insurance",
          "a phone or internet provider", "healthcare and doctors", "a game", "music", "a celebrity", "the weather", "work and colleagues",
          "school or university", "family", "a football club", "a city or country", "cars", "housing and rent", "a delivery service",
          "software and websites", "cosmetics", "furniture", "a concert or event", "pets", "prices and the economy", "a TV show",
          "an airline", "a government office", "a new law", "a hobby"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt(lang, genre, topic, length):
    return f"""Write 10 different realistic texts in {lang}. Each is a {genre} about {topic}, written by different ordinary people.
- 3 clearly POSITIVE, 3 clearly NEGATIVE, 2 NEUTRAL (factual, a plain question or statement without emotion),
  2 HARD cases (mixed feelings, sarcasm, subtle or negated sentiment, or polite criticism).
- Length: {length}. Vary tone, vocabulary and style as real people do (slang, emojis, typos or informal grammar where natural
  for this kind of text). Write naturally in {lang}, as a native speaker; do not translate from English.
- Do not mention these instructions.
Return only JSON: {{"items": [{{"text": "...", "type": "positive|negative|neutral|hard"}}, ...]}}"""


def prompt_long(lang, genre, topic, _):
    return f"""Write 4 different realistic LONG texts in {lang}, each 120-350 words. Each is a {genre} about {topic}, written by
different ordinary people with their own details (names, places, numbers, what happened).
- 1 clearly POSITIVE, 1 clearly NEGATIVE, 1 NEUTRAL (informational, a description or question without emotion),
  1 HARD case (mixed feelings that balance out, sarcasm, or criticism that turns positive at the end or the reverse).
- Write naturally in {lang}, as a native speaker; do not translate from English. Do not mention these instructions.
Return only JSON: {{"items": [{{"text": "...", "type": "positive|negative|neutral|hard"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 100)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(1300 + shard)
    LEN = ["a few words to one sentence", "one or two sentences", "two to four sentences", "a short paragraph (3-6 sentences)"]
    jobs = [(l, rng.choice(GENRES), rng.choice(TOPICS), rng.choice(LEN)) for l in LANGS for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    long = os.environ.get("LONG") == "1"
    outs = llm.chat([[{"role": "user", "content": (prompt_long if long else prompt)(*j)}] for j in jobs],
                    SamplingParams(temperature=0.95, top_p=0.95, max_tokens=3500 if long else 2500, seed=31 + shard), chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True)
    k = 0
    with open(f"{out}/gen_{'long_' if long else ''}{shard}.jsonl", "w") as f:
        for (l, g, t, _), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            for it in (j or {}).get("items", []) if isinstance(j, dict) else []:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 3 <= len(it["text"]) <= (4000 if long else 1500):
                    f.write(json.dumps(dict(text=it["text"].strip(), lang=l, genre=g, topic=t, intended=str(it.get("type", ""))),
                                       ensure_ascii=False) + "\n"); k += 1
    print("items", k, "from", len(jobs), "prompts", flush=True)
    hard_exit(0)
