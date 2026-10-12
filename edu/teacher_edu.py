"""Qwen3.8-27B as an educational-quality teacher (vLLM venv), with the FineWeb-Edu additive 0-5 rubric (HuggingFaceFW/fineweb-edu-llama3-
annotations) extended to any language, asked for the score directly; score = expected value of the next-token distribution over 0..5.
Eval:  python edu/teacher_edu.py eval OUT.json EVAL_DIR   (agreement with Llama-3-70B annotations)
Label: LABEL_IN=in.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] python edu/teacher_edu.py label   (adds 'edu', expected score)
MODEL=google/gemma-4-31b-it QUANT=fp8 runs the same rubric with an independent judge (multilingual test set)."""
import glob, json, math, os, sys

RUBRIC = """Below is an extract from a web page (it may be in any language). Evaluate whether the page has a high educational value and could be useful in an educational setting for teaching from primary school to grade school levels using the additive 5-point scoring system described below. Points are accumulated based on the satisfaction of each criterion:

- Add 1 point if the extract provides some basic information relevant to educational topics, even if it includes some irrelevant or non-academic content like advertisements and promotional material.
- Add another point if the extract addresses certain elements pertinent to education but does not align closely with educational standards. It might mix educational content with non-educational material, offering a superficial overview of potentially useful topics, or presenting information in a disorganized manner and incoherent writing style.
- Award a third point if the extract is appropriate for educational use and introduces key concepts relevant to school curricula. It is coherent though it may not be comprehensive or could include some extraneous information. It may resemble an introductory section of a textbook or a basic tutorial that is suitable for learning but has notable limitations like treating concepts that are too complex for grade school students.
- Grant a fourth point if the extract highly relevant and beneficial for educational purposes for a level not higher than grade school, exhibiting a clear and consistent writing style. It could be similar to a chapter from a textbook or a tutorial, offering substantial educational content, including exercises and solutions, with minimal irrelevant information, and the concepts aren't too advanced for grade school students. The content is coherent, focused, and valuable for structured learning.
- Bestow a fifth point if the extract is outstanding in its educational value, perfectly suited for teaching either at primary school or grade school. It follows detailed reasoning, the writing style is easy to follow and offers profound and thorough insights into the subject matter, devoid of any non-educational or complex content.

The extract:
"""


def prompt(text):
    return RUBRIC + text[:6000] + "\n\nJudge the extract in its own language; the language itself does not add or remove points. Answer with only the total score, a single digit from 0 to 5."


def score(llm, texts, max_doc_tokens=2000):
    from vllm import SamplingParams
    tk = llm.get_tokenizer()  # cut documents to max_doc_tokens teacher tokens (6,000 characters of some scripts exceed the context)
    texts = [tk.decode(ids[:max_doc_tokens]) if len(ids) > max_doc_tokens else t for t, ids in zip(texts, tk(texts, add_special_tokens=False)["input_ids"])]
    outs = llm.chat([[{"role": "user", "content": prompt(t)}] for t in texts], SamplingParams(temperature=0, max_tokens=1, logprobs=20),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    S = []
    for o in outs:
        p = [0.0] * 6
        for v in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).values():
            t = v.decoded_token.strip()
            if t in "012345" and len(t) == 1:
                p[int(t)] += math.exp(v.logprob)
        z = sum(p); S.append(sum(k * x for k, x in enumerate(p)) / z if z > 0 else float("nan"))
    return S


if __name__ == "__main__":
    import importlib.util
    import numpy as np, pandas as pd
    from vllm import LLM
    here = os.path.dirname(os.path.abspath(__file__)); sys.path.insert(0, here)
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(here, "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    kw = {"quantization": os.environ["QUANT"]} if os.environ.get("QUANT") else {}  # e.g. QUANT=fp8 for the independent judge (google/gemma-4-31b-it)
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=256,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    if sys.argv[1] == "eval":
        from metrics import f1
        out, ev = sys.argv[2:4]; res = {}
        for p in sorted(glob.glob(f"{ev}/*.parquet")):
            name = os.path.basename(p)[:-8]; df = pd.read_parquet(p); s = np.array(score(llm, df.text.tolist())); y = df.score.values
            df.assign(teacher=s).drop(columns=["text"]).to_parquet(f"{os.path.dirname(out)}/{name}_teacher.parquet")  # keep raw scores
            rk = lambda a: pd.Series(a).rank().values  # Spearman = Pearson of ranks (no scipy in the vLLM venv)
            res[name] = dict(n=len(df), spearman=float(np.corrcoef(rk(s), rk(y))[0, 1]), pearson=float(np.corrcoef(s, y)[0, 1]),
                             f1_ge3=f1(y >= 2.5, s >= 2.5), acc_round=float((np.clip(np.round(s), 0, 5) == y).mean()), mean_pred=float(np.nanmean(s)))
            print(name, res[name], flush=True)
        json.dump(res, open(out, "w"), indent=1)
    else:
        df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy(); df["edu"] = score(llm, df.text.tolist())
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True)
    _m.hard_exit(0)
