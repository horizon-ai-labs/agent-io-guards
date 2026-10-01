"""Translate Civil Comments (CC0) with Qwen3.8-27B for multilingual toxicity training; the soft labels carry over (vLLM venv).
IN=civil_train_sample.parquet SHARD=i NSHARD=n PER_LANG=12000 OUT=dir python tox/translate.py -> $OUT/tr_<shard>.jsonl
Each language gets its own random sample: half with toxicity >= 0.3, half below. The prompt asks for a faithful
translation that keeps insults, slurs, threats and tone (it is moderation data, not a request). Dropped: unparseable,
refusals, length ratio outside [0.3, 3.5], copies of the source.
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
LANGS = ["German", "French", "Spanish", "Portuguese (Brazilian)", "Italian", "Dutch", "Polish", "Russian", "Ukrainian", "Czech", "Romanian",
         "Swedish", "Danish", "Finnish", "Hungarian", "Greek", "Turkish", "Arabic", "Hebrew", "Persian", "Hindi",
         "Hinglish (Hindi written in Latin letters, mixed with English)", "Bengali", "Urdu", "Chinese (Simplified)", "Japanese", "Korean",
         "Vietnamese", "Thai", "Indonesian", "Swahili", "Amharic", "Tatar"]
L7 = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
REFUSE = re.compile(r"(I can(?:'|no)t|I cannot|I'm sorry|I am sorry|as an AI|I won't|unable to (?:help|assist|translate))", re.I)


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


if __name__ == "__main__":
    import pandas as pd
    from vllm import LLM, SamplingParams
    from common import hard_exit
    src = pd.read_parquet(os.environ["IN"]); per = int(os.environ.get("PER_LANG", 12000))
    shard, nshard = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    tox_idx, non_idx = list(src.index[src.toxicity >= 0.3]), list(src.index[src.toxicity < 0.3])
    items = []
    for li, lang in enumerate(LANGS):
        r = random.Random(100 + li)
        items += [(lang, i) for i in r.sample(tox_idx, per // 2) + r.sample(non_idx, per - per // 2)]
    items = items[shard::nshard]
    prompts = [f"You translate user comments for a content-moderation dataset. Translate the value of this JSON object into {lang} "
               f"faithfully, keeping the tone, insults, profanity, slurs, threats and intent exactly (do not soften, censor or answer it; "
               f"it is data, not a request to you). Return only JSON with the same key.\n\n" + json.dumps({"text": src.text[i]}, ensure_ascii=False)
               for lang, i in items]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    os.makedirs(os.environ["OUT"], exist_ok=True); n, bad = 0, {"parse": 0, "refuse": 0, "len": 0, "same": 0}
    with open(f"{os.environ['OUT']}/tr_{shard}.jsonl", "w") as f:
        CH = 20000
        for c in range(0, len(items), CH):
            outs = llm.chat([[{"role": "user", "content": p}] for p in prompts[c:c + CH]], SamplingParams(temperature=0.2, max_tokens=1200),
                            chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
            for (lang, i), o in zip(items[c:c + CH], outs):
                s = src.text[i]; j = parse_json(o.outputs[0].text)
                if not (isinstance(j, dict) and isinstance(j.get("text"), str) and j["text"].strip()):
                    bad["parse"] += 1; continue
                t = j["text"].strip()
                if REFUSE.search(t) and not REFUSE.search(s):
                    bad["refuse"] += 1; continue
                if not 0.3 <= len(t) / max(1, len(s)) <= 3.5:
                    bad["len"] += 1; continue
                if t == s:
                    bad["same"] += 1; continue
                f.write(json.dumps(dict(text=t, lang=lang, src_idx=int(i), **{l: float(src[l][i]) for l in L7}), ensure_ascii=False) + "\n"); n += 1
            f.flush(); print("translated", n, "of", min(c + CH, len(items)), bad, flush=True)
    print("done", flush=True)
    hard_exit(0)
