"""Qwen3.8-27B as a toxicity teacher with the Detoxify labels (vLLM venv). Each label is a yes/no question asked after the
comment (the comment comes first, so the 7 prompts per comment share a cached prefix); score = P(yes) / (P(yes) + P(no)).
Eval:  python tox/teacher_tox.py eval OUT.json EVAL_DIR      (toxicity question only; AUC per TextDetox language + Civil)
Label: LABEL_IN=in.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] python tox/teacher_tox.py label   (adds the 7 label columns)"""
import glob, json, math, os, sys

Q = {"toxicity": "Is this comment toxic, i.e. rude, disrespectful or unreasonable in a way that would make people want to leave a discussion?",
     "severe_toxicity": "Is this comment severely toxic, i.e. very hateful, aggressive or disrespectful?",
     "obscene": "Does this comment contain obscene or vulgar language (swear words, profanity)?",
     "threat": "Does this comment threaten anyone with violence or harm?",
     "insult": "Does this comment insult someone (inflammatory or demeaning toward a person or group)?",
     "identity_attack": "Does this comment attack or demean people because of their identity (ethnicity, nationality, religion, gender, sexual orientation, disability)?",
     "sexual_explicit": "Does this comment contain sexually explicit content?"}
LABELS = list(Q)


def prompt(text, label):
    return f"Comment (any language):\n<<<\n{text[:3000]}\n>>>\n\nQuestion: {Q[label]}\nAnswer with one word: yes or no."


def score(llm, texts, labels):
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": prompt(t, l)}] for t in texts for l in labels]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    P = []
    for o in outs:
        y = n = 0.0
        for v in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).values():
            t = v.decoded_token.strip().lower()
            if t in ("yes", "y"): y += math.exp(v.logprob)
            elif t in ("no", "n"): n += math.exp(v.logprob)
        P.append(y / (y + n) if y + n > 0 else 0.5)
    k = len(labels)
    return [P[i * k:(i + 1) * k] for i in range(len(texts))]


def auc(y, s):
    import numpy as np
    y = np.asarray(y); s = np.asarray(s); o = np.argsort(s); r = np.empty(len(s)); r[o] = np.arange(1, len(s) + 1)
    n1 = y.sum(); n0 = len(y) - n1
    return float((r[y == 1].sum() - n1 * (n1 + 1) / 2) / max(1, n1 * n0))


if __name__ == "__main__":
    import importlib.util
    import pandas as pd
    from vllm import LLM
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    if sys.argv[1] == "eval":
        out, ev = sys.argv[2:4]; res = {}
        for p in sorted(glob.glob(f"{ev}/tdx_*.parquet")):
            df = pd.read_parquet(p); s = [x[0] for x in score(llm, df.text.tolist(), ["toxicity"])]
            pr = [int(v >= 0.5) for v in s]; y = df.toxic.tolist()
            tp = sum(a and b for a, b in zip(pr, y)); fp = sum(a and not b for a, b in zip(pr, y)); fn = sum((not a) and b for a, b in zip(pr, y))
            res[os.path.basename(p)[:-8]] = dict(auc=auc(y, s), f1=2 * tp / max(1, 2 * tp + fp + fn)); print(os.path.basename(p), res[os.path.basename(p)[:-8]], flush=True)
        res["_tdx_mean_auc"] = sum(v["auc"] for k, v in res.items() if k.startswith("tdx_")) / sum(k.startswith("tdx_") for k in res)
        print(res["_tdx_mean_auc"], flush=True); json.dump(res, open(out, "w"), indent=1)
    else:
        df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy(); S = score(llm, df.text.tolist(), LABELS)
        for i, l in enumerate(LABELS):
            df[l] = [x[i] for x in S]
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True)
    _m.hard_exit(0)
