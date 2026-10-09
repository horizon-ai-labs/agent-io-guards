"""Spoken-style text for punctuation restoration: Qwen3.8-27B (Apache-2.0 outputs) writes edited transcripts (subtitle style,
standard punctuation) of everyday speech: talks, podcasts, interviews, meetings, lectures, vlogs, phone calls, sermons,
sports commentary, voice messages. TED/FLORES/Europarl texts are never shown to the generator.
N=... SHARD=i OUT=dir python punct/gen_spoken.py  (vLLM venv) -> $OUT/spoken_<shard>.jsonl {text, lang, genre}"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from gen_sent import LANGS, parse_json
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

GENRES = ["a public talk to a general audience (like a conference keynote)", "a podcast conversation between two hosts", "a radio interview",
          "a team meeting at work", "a university lecture", "a YouTube vlog", "a phone call between friends", "a customer support call",
          "a voice message", "a cooking show", "live sports commentary", "a political speech", "a sermon or wedding speech",
          "a TV news report with an on-the-street interview", "a teacher explaining something to a class", "a startup pitch",
          "a museum or city tour guide", "a doctor explaining a diagnosis to a patient", "a gaming livestream", "a stand-up comedy set"]
TOPICS = ["technology", "health", "climate", "history", "family life", "money", "education", "sports", "travel", "food", "science",
          "music", "work and careers", "a local news event", "relationships", "art", "politics", "a hobby", "the city", "animals"]


def prompt(lang, genre, topic):
    return f"""Write 3 different realistic transcripts of {genre} about {topic}, spoken in {lang}. Each 200-350 words.
Make it sound like real speech: spoken word order, short and long sentences, questions, asides, repetitions, "you know"-style fillers
where natural, direct address to the listener. Punctuate it the way good subtitles or an edited transcript would: full stops, commas,
question marks, colons and dashes used correctly for {lang}. Use speaker names only if there are several speakers ("Anna: ...").
Write naturally in {lang}, as a native speaker; do not translate from English. Do not mention these instructions.
Return only JSON: {{"transcripts": ["...", "...", "..."]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 4000)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(9100 + shard)
    jobs = [(rng.choice(LANGS), rng.choice(GENRES), rng.choice(TOPICS)) for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs], SamplingParams(temperature=0.9, top_p=0.95, max_tokens=4000, seed=500 + shard),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0
    with open(f"{out}/spoken_{shard}.jsonl", "w") as f:
        for (l, g, t), o in zip(jobs, outs):
            p = parse_json(o.outputs[0].text)
            for tr in (p or {}).get("transcripts", []) if isinstance(p, dict) else []:
                if isinstance(tr, str) and len(tr) > 300:
                    f.write(json.dumps(dict(text=tr.strip(), lang=l, genre=g), ensure_ascii=False) + "\n"); k += 1
    print("transcripts", k, "from", len(jobs), flush=True); hard_exit(0)
