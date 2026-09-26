"""Native-language label evals: translate MASSIVE / SIB-200 label sets and templates into each eval language (Qwen3.8-27B)
and write per-language eval files with native labels, so we can measure zero-shot classification when users write their
candidate labels in the language of the text.
python zeroshot/translate_labels.py EVAL_DIR OUT_DIR   (vLLM venv)  ->  OUT_DIR/{massive,sib200}_native_<lang>.json
"""
import json, os, re, sys

EV, OUT = sys.argv[1], sys.argv[2]
NAMES = dict(de="German", fr="French", es="Spanish", pt="Portuguese", ru="Russian", pl="Polish", tr="Turkish", ar="Arabic",
             hi="Hindi", zh="Simplified Chinese", ja="Japanese", ko="Korean", vi="Vietnamese", id="Indonesian", sw="Swahili")


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    sets = {n: json.load(open(f"{EV}/{n}.json")) for n in ["massive", "sib200"]}
    jobs = []
    for n, d in sets.items():
        for code, lang in NAMES.items():
            src = {"template": d["template"], "labels": d["labels"]}
            jobs.append((n, code, f"Translate this classification label set and hypothesis template into {lang}, as a native "
                                  f"speaker would name these categories in an app. Keep '{{}}' in the template exactly once, where "
                                  f"the label goes (adjust grammar around it if needed). Keep the labels in the same order and "
                                  f"count. Return only JSON with the same keys.\n\n{json.dumps(src, ensure_ascii=False)}"))
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.9, max_num_seqs=64,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for _, _, p in jobs], SamplingParams(temperature=0.1, max_tokens=2500),
                    chat_template_kwargs={"enable_thinking": False})
    for (n, code, _), o in zip(jobs, outs):
        j = parse_json(o.outputs[0].text); d = sets[n]
        if not (isinstance(j, dict) and isinstance(j.get("labels"), list) and len(j["labels"]) == len(d["labels"])
                and isinstance(j.get("template"), str) and j["template"].count("{}") == 1):
            print("FAILED", n, code, flush=True); continue
        items = [it for it in d["items"] if it["lang"] == code]
        json.dump(dict(template=j["template"], labels=j["labels"], items=items), open(f"{OUT}/{n}_native_{code}.json", "w"), ensure_ascii=False)
        print(n, code, j["template"], j["labels"][:4], flush=True)
