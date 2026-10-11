"""Qwen3.8-27B as a hate-speech teacher (vLLM venv): yes/no questions asked after the text (shared cached prefix);
score = P(yes) / (P(yes) + P(no)).
Eval:  python hate/teacher_hate.py eval OUT.json EVAL_DIR   (hate question on every set; target questions on MHS hateful comments)
Label: LABEL_IN=in.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] python hate/teacher_hate.py label   (adds hate + target_* columns)"""
import glob, json, math, os, sys

HATE = ("Is this hate speech? Hate speech attacks, demeans, dehumanises, stereotypes or calls for harm or exclusion against people "
        "because of a protected characteristic: race or ethnicity, nationality or migration status, religion, gender, sexual "
        "orientation, gender identity, disability or age. Not hate speech: insults or profanity aimed at an individual without "
        "targeting such a group, criticism of ideas, governments or organisations, counter-speech, and quoting or reporting hate "
        "in order to criticise it.")
GROUP = {"race": "race or ethnicity", "religion": "religion", "origin": "nationality, national origin or migration status",
         "gender": "gender (women, men, transgender or non-binary people)", "sexuality": "sexual orientation (gay, lesbian, bisexual people)",
         "age": "age (children, young or old people)", "disability": "disability (physical, mental or cognitive)"}
Q = {"hate": HATE, **{f"target_{g}": f"Does this text attack or demean people because of their {d}?" for g, d in GROUP.items()}}
LABELS = list(Q)


def prompt(text, label):
    return f"Text (any language):\n<<<\n{text[:3000]}\n>>>\n\nQuestion: {Q[label]}\nAnswer with one word: yes or no."


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


if __name__ == "__main__":
    import importlib.util
    import numpy as np, pandas as pd
    from vllm import LLM
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    if sys.argv[1] == "eval":
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__))); from metrics import auc, f1
        out, ev = sys.argv[2:4]; res = {}
        for p in sorted(glob.glob(f"{ev}/*.parquet")):
            name = os.path.basename(p)[:-8]; df = pd.read_parquet(p)
            labs = LABELS if name == "mhs" else ["hate"]
            S = np.array(score(llm, df.text.tolist(), labs)); s = S[:, 0]; y = df.hate.values
            r = dict(n=len(df), auc=auc(y, s), f1_05=f1(y, s >= 0.5), fpr_05=float((s[y == 0] >= 0.5).mean()))
            if name.startswith("mhc_"):
                r["func_acc"] = {fn: float(((s[g.index] >= 0.5) == g.hate.values.astype(bool)).mean()) for fn, g in df.groupby("functionality")}
            if name == "mhs":
                h = y == 1; r["target_auc"] = {g: auc(df[f"target_{g}"].values[h], S[h, 1 + i]) for i, g in enumerate(GROUP)}
            res[name] = r; print(name, {k: v for k, v in r.items() if k != "func_acc"}, flush=True)
        mhc = [k for k in res if k.startswith("mhc_")]
        if mhc:
            res["_mhc_mean_auc"] = float(np.mean([res[k]["auc"] for k in mhc])); res["_mhc_mean_f1"] = float(np.mean([res[k]["f1_05"] for k in mhc]))
        for key, sets in (("_real_mean_auc", ("mhs", "hatexplain", "kmhas")), ("_heldout_real_mean_auc", ("hs18", "ethos", "olidbr", "kmhas"))):
            if all(k in res for k in sets):
                res[key] = float(np.mean([res[k]["auc"] for k in sets]))
        print({k: v for k, v in res.items() if k.startswith("_")}, flush=True); json.dump(res, open(out, "w"), indent=1)
    else:
        df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy(); S = score(llm, df.text.tolist(), LABELS)
        for i, l in enumerate(LABELS):
            df[l] = [x[i] for x in S]
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True)
    _m.hard_exit(0)
