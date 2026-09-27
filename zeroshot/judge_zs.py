"""Qwen3.8-27B as an entailment judge on the zero-shot benchmarks (teacher check before any distillation).  vLLM venv.
python zeroshot/judge_zs.py OUT.json N name=path.json [name=path.json ...]
For each sampled item (seed 0, N per set) and each candidate label: P("Yes") that the text entails the set's hypothesis
template filled with the label; prediction = argmax over labels (the transformers zero-shot pipeline semantics).
"""
import json, math, os, random, sys

PROMPT = ("Text:\n<<<\n{text}\n>>>\n\nHypothesis: \"{hyp}\"\n\nIs the hypothesis true for this text? Answer with one word: Yes or No.")

def label_mode():
    """LABEL_IN=pairs.parquet (text, text_pair) SHARD=i NSHARD=n LABEL_OUT=out.parquet: adds row + qwen = P(Yes)."""
    import pandas as pd
    from vllm import LLM, SamplingParams
    df = pd.read_parquet(os.environ["LABEL_IN"], columns=["text", "text_pair"]); df["row"] = range(len(df))
    df = df.iloc[int(os.environ.get("SHARD", 0))::int(os.environ.get("NSHARD", 1))].copy()
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(text=t[:3000], hyp=h)}] for t, h in zip(df.text, df.text_pair)],
                    SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False})
    sc = []
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "yes")
        pn = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "no")
        sc.append(py / (py + pn) if py + pn else 0.5)
    df["qwen"] = sc
    df[["row", "qwen"]].to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True)


if __name__ == "__main__" and os.environ.get("LABEL_IN"):
    label_mode()
elif __name__ == "__main__":
    from vllm import LLM, SamplingParams
    out, n = sys.argv[1], int(sys.argv[2])
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    res = {}
    for spec in sys.argv[3:]:
        name, path = spec.split("=", 1)
        d = json.load(open(path)); items = d["items"][:]; random.Random(0).shuffle(items); items = items[:n]
        prompts = [PROMPT.format(text=it["text"][:3000], hyp=d["template"].replace("{}", l, 1)) for it in items for l in d["labels"]]
        outs = llm.chat([[{"role": "user", "content": p}] for p in prompts], SamplingParams(temperature=0, max_tokens=1, logprobs=20),
                        chat_template_kwargs={"enable_thinking": False})
        sc = []
        for o in outs:
            lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
            py = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "yes")
            pn = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "no")
            sc.append(py / (py + pn) if py + pn else 0.0)
        L = len(d["labels"]); correct = 0; per = {}
        for i, it in enumerate(items):
            s = sc[i * L:(i + 1) * L]; ok = int(max(range(L), key=lambda k: s[k]) == it["gold"]); correct += ok
            per.setdefault(it["lang"], []).append(ok)
        res[name] = dict(n=len(items), acc=correct / len(items), per_lang={k: sum(v) / len(v) for k, v in per.items()})
        print(name, round(res[name]["acc"], 3), flush=True)
        json.dump(res, open(out, "w"), indent=1)
