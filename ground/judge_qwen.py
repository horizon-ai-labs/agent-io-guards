"""Qwen3.8-27B as a groundedness judge (candidate teacher for hallucination-guard distillation).  vLLM venv.
python ground/judge_qwen.py OUT.json DIR [DIR ...]      (DIR: *.parquet with text (document), text_pair (claim/response), label)
Score = P("Yes") / (P("Yes") + P("No")) for the first answer token; metrics: balanced accuracy at 0.5 and ROC AUC.
Also usable as a labeller:  LABEL_IN=pairs.parquet LABEL_OUT=out.parquet python ground/judge_qwen.py - -
"""
import glob, json, math, os, sys

PROMPT = ("Document:\n<<<\n{doc}\n>>>\n\nStatement:\n<<<\n{claim}\n>>>\n\nIs every part of the statement supported by the document? "
          "Judge only by the document, not by world knowledge; a statement that adds, changes or contradicts any detail is not "
          "supported. Answer with one word: Yes or No.")


def scores(llm, tok, pairs):
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": PROMPT.format(doc=d[:24000], claim=c[:4000])}] for d, c in pairs]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False})
    res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        py = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "yes")
        pn = sum(math.exp(v.logprob) for v in lp.values() if v.decoded_token.strip().lower() == "no")
        res.append(py / (py + pn) if py + pn > 0 else 0.5)
    return res


if __name__ == "__main__":
    import pandas as pd
    from vllm import LLM
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=16384, gpu_memory_utilization=0.92, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0})
    if os.environ.get("LABEL_IN"):
        df = pd.read_parquet(os.environ["LABEL_IN"]); df["row"] = range(len(df))
        sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy()
        df["judge"] = scores(llm, None, list(zip(df.text, df.text_pair)))
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df)); sys.exit(0)
    import numpy as np

    def balanced_accuracy_score(y, pr):   # sklearn is not in the vLLM venv
        y, pr = np.asarray(y), np.asarray(pr)
        return float(np.mean([(pr[y == c] == c).mean() for c in (0, 1) if (y == c).any()]))

    def roc_auc_score(y, s):   # Mann-Whitney U with average ranks
        y, s = np.asarray(y), np.asarray(s, dtype=float)
        order = s.argsort(); ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
        for v in np.unique(s):
            m = s == v; ranks[m] = ranks[m].mean()
        n1 = (y == 1).sum(); n0 = len(y) - n1
        return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    out, res = sys.argv[1], {}
    for d in sys.argv[2:]:
        for p in sorted(glob.glob(f"{d}/*.parquet")):
            df = pd.read_parquet(p); s = scores(llm, None, list(zip(df.text, df.text_pair)))
            y = df.label.values; pr = [x > 0.5 for x in s]
            res[os.path.basename(p)[:-8]] = dict(n=len(df), bacc=float(balanced_accuracy_score(y, pr)),
                                                  auc=float(roc_auc_score(y, s)) if 0 < y.mean() < 1 else None)
            print(os.path.basename(p), res[os.path.basename(p)[:-8]], flush=True)
    agg = [v["bacc"] for k, v in res.items() if k.startswith("aggrefact_")]
    res["_aggrefact_mean_bacc"] = sum(agg) / max(1, len(agg)); print("AggreFact mean bacc", res["_aggrefact_mean_bacc"])
    json.dump(res, open(out, "w"), indent=1)
