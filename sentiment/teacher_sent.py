"""Qwen3.8-27B as a sentiment teacher (vLLM venv).
Eval:   python sentiment/teacher_sent.py OUT.json EVAL_DIR
Label:  LABEL_IN=texts.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] python sentiment/teacher_sent.py - -
Scores = first-answer-token probabilities of negative / neutral / positive, renormalised (soft labels p_neg, p_neu, p_pos).
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import C3, hard_exit, load_evals, score, summarize

PROMPT = ("Text:\n<<<\n{text}\n>>>\n\nWhat is the overall sentiment the author expresses in this text?\n"
          "- positive: approval, praise, satisfaction, joy, gratitude, excitement\n"
          "- negative: disapproval, complaint, anger, sadness, disappointment, criticism\n"
          "- neutral: factual or informational, a question or request without emotion, or no clear sentiment either way\n"
          "Answer with one word: positive, negative or neutral.")
# v2: mixed / balanced texts are neutral (the tweets and Amazon 3-star conventions); env PROMPT_V=2
PROMPT2 = PROMPT.replace("or no clear sentiment either way", "no clear sentiment, or a mix of positive and negative points "
                         "that roughly balance out (e.g. \"good quality but too expensive\", an average experience)")
if os.environ.get("PROMPT_V") == "2":
    PROMPT = PROMPT2


def probs(llm, texts):
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": PROMPT.format(text=t[:6000])}] for t in texts]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False})
    res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        p = [0.0, 0.0, 0.0]
        for v in lp.values():
            t = v.decoded_token.strip().lower()
            for i, c in enumerate(C3):
                if len(t) >= 3 and c.startswith(t):
                    p[i] += math.exp(v.logprob)
        s = sum(p)
        res.append([x / s for x in p] if s > 0 else [1 / 3] * 3)
    return res


if __name__ == "__main__":
    import pandas as pd
    from vllm import LLM
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0})
    if os.environ.get("LABEL_IN"):
        df = pd.read_parquet(os.environ["LABEL_IN"])
        sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy()
        p = probs(llm, df.text.tolist())
        df["p_neg"], df["p_neu"], df["p_pos"] = zip(*p)
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True); hard_exit(0)
    out, res = sys.argv[1], {}
    only = os.environ.get("ONLY", "")   # e.g. "tweets,amazon"
    for name, df in load_evals(sys.argv[2]).items():
        if only and not name.startswith(tuple(only.split(","))):
            continue
        res[name] = score(df, probs(llm, df.text.tolist())); print(name, res[name], flush=True)
    res.update(summarize(res)); print({k: v for k, v in res.items() if k.startswith("_")}, flush=True)
    json.dump(res, open(out, "w"), indent=1)
    hard_exit(0)
