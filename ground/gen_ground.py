"""Synthetic groundedness data (Qwen3.8-27B via vLLM) over FineWeb-Edu / FineWeb-2 passages (ODC-BY).

For each passage Qwen writes, in the passage's language: a user task (question / summary request / extraction),
one fully supported response, and several unsupported variants of distinct error types. Output jsonl rows:
  {passage, lang, task, responses: [{text, label (1 supported / 0 unsupported), type}]}
"""
import json, os, random, re, sys, time


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


OUT = os.environ.get("OUT", "."); SHARD = int(os.environ.get("SHARD", 0)); N = int(os.environ.get("N", 200))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")
rng = random.Random(9000 + SHARD)
LANGS = {"en": None, "deu_Latn": "German", "fra_Latn": "French", "spa_Latn": "Spanish", "por_Latn": "Portuguese",
         "ita_Latn": "Italian", "nld_Latn": "Dutch", "pol_Latn": "Polish", "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian",
         "tur_Latn": "Turkish", "arb_Arab": "Arabic", "hin_Deva": "Hindi", "cmn_Hani": "Chinese", "jpn_Jpan": "Japanese",
         "kor_Hang": "Korean", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian", "tha_Thai": "Thai", "ces_Latn": "Czech",
         "swe_Latn": "Swedish", "fas_Arab": "Persian", "heb_Hebr": "Hebrew", "ron_Latn": "Romanian", "ell_Grek": "Greek",
         "hun_Latn": "Hungarian", "dan_Latn": "Danish", "fin_Latn": "Finnish", "ben_Beng": "Bengali"}
TASKS = ["a specific factual question answerable from the passage", "a request to summarize the passage in 2-4 sentences",
         "a question that requires combining two facts from the passage", "a request to list key facts / numbers from it",
         "a 'why' or 'how' question the passage explains", "a request to explain it simply to a 12-year-old"]
ERRORS = ["entity swap (a name, place or organization replaced by a plausible wrong one)",
          "number / date / unit changed", "an added claim that sounds plausible but is not in the passage",
          "contradiction of a stated fact", "over-generalization or wrong causal claim", "invented quote, statistic or source",
          "mostly faithful, but exactly one sentence is unsupported"]


def prompt(passage, lang, task):
    return f"""You create evaluation data for a model that checks whether an AI response is grounded in a source document.

Source document:
<<<
{passage}
>>>

Write in {lang}. First write a user request: {task}.
Then write:
- "supported": a helpful response to the request where EVERY claim is supported by the source document.
- "unsupported": 3 responses of similar length and style, each containing a different subtle error of these kinds:
  {"; ".join(rng.sample(ERRORS, 3))}. Errors must be subtle and fluent, as real LLM hallucinations are.
Do not mention the source document or the errors inside the responses.

Return only JSON: {{"request": "...", "supported": "...", "unsupported": [{{"type": "...", "text": "..."}}]}}"""


def load_passages(n):
    from datasets import load_dataset
    out = []
    codes = list(LANGS)
    per = max(1, n // len(codes))
    for code in codes:
        name = LANGS[code]
        if code == "en":
            ds = load_dataset("HuggingFaceFW/fineweb-edu", "sample-10BT", split="train", streaming=True)
            name, k = "English", per * 6
        elif name is None:
            continue
        else:
            ds = load_dataset("HuggingFaceFW/fineweb-2", code, split="train", streaming=True)
            k = per
        skip = SHARD * k * 3
        got = 0
        for i, r in enumerate(ds):
            if i < skip:
                continue
            t = r["text"].strip()
            if not 600 <= len(t) <= 6000:
                continue
            # a coherent excerpt of up to ~2500 characters
            if len(t) > 2500:
                s = t.find("\n", rng.randint(0, len(t) - 2500))
                t = t[s + 1: s + 1 + 2500] if s >= 0 else t[:2500]
            out.append((t, name)); got += 1
            if got >= k:
                break
    rng.shuffle(out)
    return out[:n]


def prep():
    """Run with a Python that has `datasets` (not the vLLM venv): python ground/gen_ground.py prep OUTFILE"""
    json.dump(load_passages(N), open(sys.argv[2], "w"), ensure_ascii=False)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "prep":
    prep()
elif __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    passages = [tuple(x) for x in json.load(open(os.environ["PASSAGES"]))] if os.environ.get("PASSAGES") else load_passages(N)
    print("passages", len(passages), flush=True)
    jobs = [(p, l, rng.choice(TASKS)) for p, l in passages]
    llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.8, top_p=0.95, max_tokens=2500, seed=500 + SHARD)
    t = time.time()
    outs = llm.chat([[{"role": "user", "content": prompt(p, l, task)}] for p, l, task in jobs], sp,
                    chat_template_kwargs={"enable_thinking": False})
    print(f"generated {len(outs)} in {time.time()-t:.0f}s", flush=True)
    ok = 0
    with open(f"{OUT}/ground_{SHARD}.jsonl", "w") as f:
        for (p, l, task), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            if not (isinstance(j, dict) and isinstance(j.get("supported"), str) and isinstance(j.get("unsupported"), list)):
                continue
            resp = [dict(text=j["supported"], label=1, type="supported")]
            resp += [dict(text=u["text"], label=0, type=u.get("type", "")) for u in j["unsupported"]
                     if isinstance(u, dict) and isinstance(u.get("text"), str) and len(u["text"]) > 20]
            if len(resp) < 2:
                continue
            f.write(json.dumps(dict(passage=p, lang=l, task=j.get("request", ""), responses=resp), ensure_ascii=False) + "\n")
            ok += 1
    print(f"ok {ok}/{len(jobs)}", flush=True)
