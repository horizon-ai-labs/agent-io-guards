"""Synthetic multilingual financial texts for financial-sentiment training, Qwen3.8-27B via vLLM (Apache-2.0 outputs).
Each prompt asks for 10 texts in one language, genre and topic: 3 good news for investors, 3 bad news, 3 neutral (factual),
1 hard (mixed: revenue up but margin down, a loss that narrowed, a beat with weak guidance). Company names are invented or
generic. Labels come later from the teacher (sentiment/teacher_sent.py PROMPT_V=fin), not from the generator. No benchmark
items are shown to the generator.
N=... SHARD=i OUT=dir python fin/gen_fin.py   (vLLM venv) -> $OUT/gen_<shard>.jsonl {text, lang, genre, topic, intended}
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from common import hard_exit
from gen_sent import LANGS, parse_json

GENRES = ["financial news headline", "sentence from a financial news article", "short paragraph from a business news article",
          "sentence from a company press release", "excerpt from a quarterly earnings report", "sentence from an analyst note",
          "stock market post on social media (with $cashtags where natural)", "post in an investor forum", "comment from a retail investor",
          "sentence from a central bank or economic news report", "crypto market news or post", "market wrap-up sentence (end of day)",
          "headline from a local business newspaper", "message in an investors' chat group"]
TOPICS = ["quarterly earnings of a listed company", "revenue and profit guidance", "a merger or acquisition", "layoffs or hiring",
          "an IPO", "dividends and share buybacks", "a credit rating change", "a lawsuit or regulatory fine", "a new product launch",
          "a CEO or management change", "a bank", "an airline", "a carmaker", "a retailer", "a tech company", "a pharmaceutical company",
          "an energy or oil company", "a mining company", "real estate and housing prices", "interest rates and the central bank",
          "inflation", "unemployment and the labour market", "GDP growth and recession fears", "the national stock index",
          "currency exchange rates", "government bonds and yields", "commodities (gold, oil, wheat)", "cryptocurrencies",
          "a startup funding round", "a bankruptcy or restructuring", "exports, tariffs and trade", "a supply chain disruption",
          "an insurance company", "a telecom operator", "tourism and hotels", "agriculture and food prices"]


# SOCIAL=1 (v2): only investor social-media styles (FiQA/TFNS-like), English-heavy language mix
SOCIAL_GENRES = ["stock market post on social media with $cashtags", "short tweet by a trader about a position (calls, puts, buys, sells)",
                 "post in an investor forum giving an opinion on a stock", "comment from a retail investor reacting to a price move",
                 "financial news headline mentioning a ticker", "tweet sharing a financial news link with a short comment",
                 "message in an investors' chat group", "short opinion headline (e.g. 'Why I'd buy ...', '3 reasons to avoid ...')"]


def prompt(lang, genre, topic, length):
    return f"""Write 10 different realistic texts in {lang}. Each is a {genre} about {topic}. Use invented or generic company
names and realistic local details (currency, stock exchange, places) for {lang}-speaking readers.
- 3 clearly GOOD news for investors (e.g. rising profits, strong demand, an upgrade), 3 clearly BAD news (e.g. losses, falling
  prices, layoffs, a downgrade), 3 NEUTRAL (factual information without a clear positive or negative implication: schedules,
  appointments, descriptions, plain figures without comparison), 1 HARD case (mixed or subtle: revenue up but margin down, a loss
  that narrowed, results above expectations but weak guidance, a cautious outlook).
- Length: {length}. Vary style and vocabulary as real writers do. Write naturally in {lang}, as a native speaker; do not translate
  from English. Do not mention these instructions.
Return only JSON: {{"items": [{{"text": "...", "type": "positive|negative|neutral|hard"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 100)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(2600 + shard)
    LEN = ["a headline-like phrase or one short sentence", "one sentence", "one or two sentences", "two to four sentences"]
    if os.environ.get("SOCIAL") == "1":
        langs = ["English"] * 40 + LANGS
        jobs = [(l, rng.choice(SOCIAL_GENRES), rng.choice(TOPICS), rng.choice(LEN[:3])) for l in langs for _ in range(n)]
    else:
        jobs = [(l, rng.choice(GENRES), rng.choice(TOPICS), rng.choice(LEN)) for l in LANGS for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs],
                    SamplingParams(temperature=0.95, top_p=0.95, max_tokens=2500, seed=71 + shard), chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True)
    k = 0
    with open(f"{out}/gen_{'social_' if os.environ.get('SOCIAL') == '1' else ''}{shard}.jsonl", "w") as f:
        for (l, g, t, _), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            for it in (j or {}).get("items", []) if isinstance(j, dict) else []:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 3 <= len(it["text"]) <= 1500:
                    f.write(json.dumps(dict(text=it["text"].strip(), lang=l, genre=g, topic=t, intended=str(it.get("type", ""))),
                                       ensure_ascii=False) + "\n"); k += 1
    print("items", k, "from", len(jobs), "prompts", flush=True)
    hard_exit(0)
