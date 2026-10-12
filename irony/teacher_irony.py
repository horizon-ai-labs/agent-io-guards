"""Qwen3.8-27B as an irony/sarcasm teacher (vLLM venv): one yes/no question; score = P(yes) / (P(yes) + P(no)).
The context (the post a reply answers), when given, comes first.
Eval:  python irony/teacher_irony.py eval OUT.json EVAL_DIR
Label: LABEL_IN=in.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] python irony/teacher_irony.py label   (adds column 'irony')"""
import glob, json, math, os, sys

Q = ("Is the text ironic or sarcastic? That is: does it mean something different from, often the opposite of, what it literally "
     "says (for example mocking praise, feigned agreement, exaggeration or understatement meant to be noticed)? Sincere statements, "
     "plain jokes, insults and complaints that mean what they say are not ironic.")


def prompt(text, context):
    c = f"Context: this text is a reply to the following post.\n<<<\n{context[:2000]}\n>>>\n\n" if context else ""
    return f"{c}Text (any language):\n<<<\n{text[:2000]}\n>>>\n\nQuestion: {Q}\nAnswer with one word: yes or no."


def score(llm, texts, contexts):
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": prompt(t, c)}] for t, c in zip(texts, contexts)]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    P = []
    for o in outs:
        y = n = 0.0
        for v in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).values():
            t = v.decoded_token.strip().lower()
            if t in ("yes", "y"): y += math.exp(v.logprob)
            elif t in ("no", "n"): n += math.exp(v.logprob)
        P.append(y / (y + n) if y + n > 0 else 0.5)
    return P


if __name__ == "__main__":
    import importlib.util
    import numpy as np, pandas as pd
    from vllm import LLM
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(here, "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0})
    if sys.argv[1] == "eval":
        from metrics import auc, f1
        out, ev = sys.argv[2:4]; res = {}
        for p in sorted(glob.glob(f"{ev}/*.parquet")):
            name = os.path.basename(p)[:-8]; df = pd.read_parquet(p); y = df.label.values
            s = np.array(score(llm, df.text.tolist(), df.context.fillna("").tolist()))
            r = dict(n=len(df), auc=auc(y, s), f1_05=f1(y, s >= 0.5))
            if (df.context.fillna("") != "").any():  # also without context
                s0 = np.array(score(llm, df.text.tolist(), [""] * len(df))); r.update(auc_noctx=auc(y, s0), f1_05_noctx=f1(y, s0 >= 0.5))
            res[name] = r; print(name, r, flush=True)
        mp = [k for k in res if k.startswith("mpico_")]
        res["_mpico_mean_auc"] = float(np.mean([res[k]["auc"] for k in mp])); res["_mpico_mean_auc_noctx"] = float(np.mean([res[k]["auc_noctx"] for k in mp]))
        res["_tweets_mean_auc"] = float(np.mean([res[k]["auc"] for k in ("semeval18", "isarcasm", "ar_sarcasm", "ko_sarcasm") if k in res]))
        print({k: v for k, v in res.items() if k.startswith("_")}, flush=True); json.dump(res, open(out, "w"), indent=1)
    else:
        df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy(); df["irony"] = score(llm, df.text.tolist(), df.get("context", pd.Series([""] * len(df))).fillna("").tolist())
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True)
    _m.hard_exit(0)
