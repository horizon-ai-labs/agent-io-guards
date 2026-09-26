"""Translate the label sets (+ hypothesis templates) of non-English synthetic zero-shot items into the item's language, so
training can include native-language labels (users often write labels in the language of the text).
python zeroshot/translate_labelsets.py OUT.jsonl GEN_DIR [GEN_DIR ...]   (vLLM venv)
Output lines: {"lang", "labels" (English, as in the data), "template", "native_labels", "native_template"}
"""
import glob, json, os, re, sys

out, gens = sys.argv[1], sys.argv[2:]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    keys = {}
    for g in gens:
        for p in glob.glob(f"{g}/zs_*.jsonl") + glob.glob(f"{g}/zs2_*.jsonl"):
            if "_99." in p:
                continue
            for line in open(p):
                j = json.loads(line)
                if j["lang"] == "English" or not j["labels"]:
                    continue
                k = (j["lang"], tuple(j["labels"]), j.get("template", "") or "This example is {}.")
                keys[k] = 1
    keys = list(keys)
    print("unique label sets", len(keys), flush=True)
    prompts = [f"Translate this classification label set and hypothesis template into {lang}, as a native speaker would name "
               f"these categories. Keep '{{}}' in the template exactly once, where a label goes (adjust grammar around it). Keep "
               f"the labels in the same order and count. Return only JSON with the same keys.\n\n"
               + json.dumps({"template": tpl, "labels": list(labels)}, ensure_ascii=False) for lang, labels, tpl in keys]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SamplingParams(temperature=0.2, max_tokens=1200),
                    chat_template_kwargs={"enable_thinking": False})
    n = 0
    with open(out, "w") as f:
        for (lang, labels, tpl), o in zip(keys, outs):
            j = parse_json(o.outputs[0].text)
            if not (isinstance(j, dict) and isinstance(j.get("labels"), list) and len(j["labels"]) == len(labels)
                    and all(isinstance(x, str) and x.strip() for x in j["labels"]) and isinstance(j.get("template"), str)
                    and j["template"].count("{}") == 1):
                continue
            f.write(json.dumps(dict(lang=lang, labels=list(labels), template=tpl, native_labels=j["labels"], native_template=j["template"]),
                               ensure_ascii=False) + "\n"); n += 1
    print("translated", n, "of", len(keys), flush=True)
