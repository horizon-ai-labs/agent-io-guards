"""Stage B of the compression eval: an LLM reader answers each question from each compressed context; SQuAD-style F1 / EM against the
gold answers (max over answers; character tokens for zh/ja/th, whitespace tokens otherwise; lowercased, punctuation removed).
QA=qa.parquet CTX_FILE=compressed.parquet OUT=result.json [MODEL=...] python compress/read_qa.py   (vLLM venv)
Reader: an independent model family from the teacher (default google/gemma-4-31b-it)."""
import collections, json, os, re, string, sys, unicodedata
import pandas as pd
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit   # kills the vLLM engine child too

PROMPT = """Answer the question using the context below. Some words may have been removed from the context. Reply with only the answer, a short phrase taken from the context if possible, in the language of the question.

Context:
{ctx}

Question: {q}"""
CHAR = {"zh", "ja", "th"}


def norm(s, lang):
    s = unicodedata.normalize("NFKC", s).lower()
    s = "".join(" " if unicodedata.category(c)[0] in "PS" else c for c in s)
    if lang == "en":
        s = re.sub(r"\b(a|an|the)\b", " ", s)
    return [c for c in s if not c.isspace()] if lang in CHAR else s.split()


def f1(pred, gold, lang):
    p, g = norm(pred, lang), norm(gold, lang)
    if not p or not g:
        return float(p == g), float(p == g)
    common = collections.Counter(p) & collections.Counter(g); n = sum(common.values())
    if n == 0:
        return 0.0, 0.0
    pr, rc = n / len(p), n / len(g)
    return 2 * pr * rc / (pr + rc), float(p == g)


def summarize(df):
    by = df.groupby(["system", "rate", "set", "lang"]).agg(f1=("f1", "mean"), em=("em", "mean"), kept_chars=("kept_chars", "mean")).reset_index()
    out = {}
    for (s, r), g in by.groupby(["system", "rate"]):
        out[f"{s}@{r}"] = dict(system=s, rate=r, sets={st: dict(f1=float(x.f1.mean()), em=float(x.em.mean()), kept_chars=float(x.kept_chars.mean()),
                                                                per_lang={l: round(v, 4) for l, v in zip(x.lang, x.f1)}) for st, x in g.groupby("set")})
        out[f"{s}@{r}"]["mean_f1"] = float(g.f1.mean())
    return out


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    qa = pd.read_parquet(os.environ["QA"]); cc = pd.read_parquet(os.environ["CTX_FILE"])
    df = cc.merge(qa[["set", "lang", "question", "answers"]], left_on="qid", right_index=True)
    llm = LLM(os.environ.get("MODEL", "google/gemma-4-31b-it"), max_model_len=8192, gpu_memory_utilization=0.9, max_num_seqs=256,
              quantization=os.environ.get("QUANT") or None, limit_mm_per_prompt={"image": 0, "video": 0})
    msgs = [[{"role": "user", "content": PROMPT.format(ctx=c, q=q)}] for c, q in zip(df.context, df.question)]
    outs = llm.chat(msgs, SamplingParams(temperature=0.0, max_tokens=64), use_tqdm=False)
    df["pred"] = [o.outputs[0].text.strip().split("\n")[0] for o in outs]
    sc = [max((f1(p, g, l) for g in a), default=(0.0, 0.0)) for p, a, l in zip(df.pred, df.answers, df.lang)]
    df["f1"], df["em"] = [x[0] for x in sc], [x[1] for x in sc]
    df.drop(columns=["context"]).to_parquet(os.environ["OUT"].replace(".json", "_rows.parquet"), index=False)
    res = summarize(df); json.dump(res, open(os.environ["OUT"], "w"), indent=1)
    for k, v in sorted(res.items(), key=lambda kv: -kv[1]["mean_f1"]):
        print(k, round(v["mean_f1"], 4), {s: (round(x["f1"], 4), round(x["kept_chars"], 3)) for s, x in v["sets"].items()}, flush=True)
    hard_exit(0)
