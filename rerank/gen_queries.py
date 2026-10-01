"""Search queries for passages, Qwen3.8-27B via vLLM (Apache-2.0 outputs).
IN=passages.parquet SHARD=i NSHARD=n OUT=dir python rerank/gen_queries.py -> $OUT/q_<shard>.jsonl {pid, query, qtype, qlang}
Per passage: one natural-language question and one short keyword-style query, in the passage's language; for 15% of
non-English passages one more query in English (cross-lingual search). Queries must be answerable by the passage but
should not copy long phrases from it.
"""
import json, os, random, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prompt(text, xl):
    extra = ('\n- "english": a natural question in English that this passage answers (for cross-lingual search)' if xl else "")
    return f"""Passage:
<<<
{text}
>>>

Write search queries that a user might type into a search engine and for which this passage is a good result.
- "question": a natural question in the same language as the passage, answered by the passage
- "keywords": a short keyword-style query (2-6 words) in the same language as the passage{extra}
Do not copy long phrases from the passage; use the words a searcher would use. Return only JSON with these keys."""


if __name__ == "__main__":
    import pandas as pd
    from vllm import LLM, SamplingParams
    from common import hard_exit
    df = pd.read_parquet(os.environ["IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    df = df.iloc[sh::ns]
    rng = random.Random(11 + sh)
    xl = [l != "eng_Latn" and rng.random() < 0.15 for l in df.lang]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(t, x)}] for t, x in zip(df.text, xl)],
                    SamplingParams(temperature=0.7, top_p=0.95, max_tokens=300, seed=5 + sh), chat_template_kwargs={"enable_thinking": False})
    os.makedirs(os.environ["OUT"], exist_ok=True); n = 0
    with open(f"{os.environ['OUT']}/q_{sh}.jsonl", "w") as f:
        for pid, lang, o in zip(df.pid, df.lang, outs):
            j = parse_json(o.outputs[0].text)
            if not isinstance(j, dict):
                continue
            for qt in ["question", "keywords", "english"]:
                q = j.get(qt)
                if isinstance(q, str) and 3 <= len(q.strip()) <= 300:
                    f.write(json.dumps(dict(pid=int(pid), query=q.strip(), qtype=qt, qlang="eng_Latn" if qt == "english" else lang), ensure_ascii=False) + "\n"); n += 1
    print("queries", n, "for", len(df), "passages", flush=True)
    hard_exit(0)
