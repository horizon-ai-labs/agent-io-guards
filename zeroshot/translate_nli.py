"""Multilingual NLI training data: MultiNLI (OANC / CC-BY-SA parts) and WANLI (CC-BY-4.0) pairs machine-translated
by Qwen3.8-27B. Each language gets different source pairs. Premise and hypothesis are translated together.

prep (python with datasets):  python zeroshot/translate_nli.py prep OUT.json      (N per language, env)
run  (vLLM venv):             ITEMS=OUT.json SHARD=i NSHARD=k OUT=dir python zeroshot/translate_nli.py
Output: $OUT/nli_tr_<shard>.jsonl  {premise, hypothesis, premise_en, hypothesis_en, label (entailment|neutral|contradiction), lang, source}
"""
import json, os, random, re, sys

N = int(os.environ.get("N", 5000))
LANGS = ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian", "Ukrainian", "Czech",
         "Turkish", "Arabic", "Persian", "Hebrew", "Hindi", "Bengali", "Chinese (Simplified)", "Japanese", "Korean",
         "Vietnamese", "Indonesian", "Thai", "Swahili", "Greek"]


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def prep():
    from datasets import load_dataset
    rng = random.Random(7)
    L = {0: "entailment", 1: "neutral", 2: "contradiction"}
    mnli = [(r["premise"], r["hypothesis"], L[r["label"]], "mnli") for r in load_dataset("nyu-mll/multi_nli", split="train") if r["label"] in L]
    wanli = [(r["premise"], r["hypothesis"], r["gold"], "wanli") for r in load_dataset("alisawuffles/WANLI", split="train")]
    rng.shuffle(mnli); rng.shuffle(wanli)
    items, im, iw = [], 0, 0
    for lang in LANGS:
        for k in range(N):
            if k % 10 < 7:
                p, h, l, s = mnli[im]; im += 1
            else:
                p, h, l, s = wanli[iw]; iw += 1
            items.append(dict(premise_en=p, hypothesis_en=h, label=l, lang=lang, source=s))
    json.dump(items, open(sys.argv[2], "w"), ensure_ascii=False)
    print("items", len(items), "mnli used", im, "wanli used", iw)


if __name__ == "__main__" and len(sys.argv) > 1 and sys.argv[1] == "prep":
    prep()
elif __name__ == "__main__":
    from vllm import LLM, SamplingParams
    shard, nshard = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    items = json.load(open(os.environ["ITEMS"]))[shard::nshard]
    prompts = [f"Translate the values of this JSON object into {it['lang']}. Translate naturally and faithfully: keep the "
               f"meaning, names and numbers exactly, do not add or remove information. Return only the translated JSON "
               f"with the same keys.\n\n" + json.dumps({"premise": it["premise_en"], "hypothesis": it["hypothesis_en"]}, ensure_ascii=False)
               for it in items]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92,
              max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SamplingParams(temperature=0.2, max_tokens=1000),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(os.environ["OUT"], exist_ok=True)
    n = 0
    with open(f"{os.environ['OUT']}/nli_tr_{shard}.jsonl", "w") as f:
        for it, o in zip(items, outs):
            j = parse_json(o.outputs[0].text)
            if isinstance(j, dict) and isinstance(j.get("premise"), str) and isinstance(j.get("hypothesis"), str) and j["premise"].strip() and j["hypothesis"].strip():
                f.write(json.dumps(dict(it, premise=j["premise"].strip(), hypothesis=j["hypothesis"].strip()), ensure_ascii=False) + "\n"); n += 1
    print(f"translated {n} of {len(items)}", flush=True)
