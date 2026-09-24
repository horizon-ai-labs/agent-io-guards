"""Multilingual groundedness eval: HaluEval (Apache-2.0) QA + dialogue items machine-translated by Qwen3.8-27B.

Each item (knowledge, question/history, right, hallucinated) is translated as one JSON object so the pair stays
consistent; labels are inherited. Output: $OUT/halueval_<lang>.parquet with text / text_pair / label.
"""
import json, os, random, re, sys, time

OUT = os.environ.get("OUT", "."); N = int(os.environ.get("N", 400))
LANGS = {"de": "German", "es": "Spanish", "zh": "Simplified Chinese", "ja": "Japanese", "ar": "Arabic", "hi": "Hindi"}


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prep():
    from datasets import load_dataset
    rng = random.Random(42)
    items = []
    for cfg, kf, qf, rf, hf in [("qa", "knowledge", "question", "right_answer", "hallucinated_answer"),
                                ("dialogue", "knowledge", "dialogue_history", "right_response", "hallucinated_response")]:
        ds = list(load_dataset("pminervini/HaluEval", cfg, split="data"))
        rng.shuffle(ds)
        # skip the first 1000 (those are the English eval items) to keep the translated set disjoint
        for r in ds[1000: 1000 + N // 2]:
            items.append(dict(task=cfg, knowledge=r[kf], question=r[qf], right=r[rf], hallucinated=r[hf]))
    json.dump(items, open(sys.argv[2], "w"), ensure_ascii=False)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "prep":
    prep()
elif __name__ == "__main__":
    import pandas as pd
    from vllm import LLM, SamplingParams
    items = json.load(open(os.environ["ITEMS"]))
    jobs = []
    for code, lang in LANGS.items():
        for it in items:
            src = {k: it[k] for k in ("knowledge", "question", "right", "hallucinated")}
            jobs.append((code, it, f"Translate every value of this JSON object into {lang}. Keep names of people and "
                                   f"places recognizable, keep numbers exactly, do not add or remove information. "
                                   f"Return only the translated JSON with the same keys.\n\n{json.dumps(src, ensure_ascii=False)}"))
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=8192, gpu_memory_utilization=0.92,
              max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for _, _, p in jobs], SamplingParams(temperature=0.2, max_tokens=2500),
                    chat_template_kwargs={"enable_thinking": False})
    rows = {c: [] for c in LANGS}
    for (code, it, _), o in zip(jobs, outs):
        j = parse_json(o.outputs[0].text)
        if not (isinstance(j, dict) and all(isinstance(j.get(k), str) for k in ("knowledge", "question", "right", "hallucinated"))):
            continue
        ctx = j["knowledge"] + "\n\n" + j["question"]
        rows[code] += [dict(text=ctx, text_pair=j["right"], label=1), dict(text=ctx, text_pair=j["hallucinated"], label=0)]
    os.makedirs(OUT, exist_ok=True)
    for code, r in rows.items():
        pd.DataFrame(r).to_parquet(f"{OUT}/halueval_{code}.parquet")
        print(code, len(r), flush=True)
