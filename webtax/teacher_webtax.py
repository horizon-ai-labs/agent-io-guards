"""LLM labeller for the multilingual web topic & format classifier (vLLM venv). Prompts, the 24 topics and the 24 formats are those of
WebOrganizer (github.com/CodeCreator/WebOrganizer, Apache-2.0; webtax/taxonomies/*.yaml), used without the URL and without few-shot
demonstrations, with a note that pages may be in any language. Score = probability of each answer letter (A-X) -> soft labels.
LABEL_IN=in.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] [MODEL=google/gemma-4-31b-it QUANT=fp8] python webtax/teacher_webtax.py
adds topic_<slug> and format_<slug> columns (24 each)."""
import math, os, re, sys
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
TAX = {t: yaml.safe_load(open(os.path.join(HERE, "taxonomies", f"{t}.yaml"))) for t in ("topics", "formats")}
LET = "ABCDEFGHIJKLMNOPQRSTUVWX"
slug = lambda c: re.sub(r"[^a-z0-9]+", "_", c.split("\n")[0].lower()).strip("_")
NAMES = {t: [slug(c) for c in TAX[t]["choices"]] for t in TAX}
LABELS = [f"topic_{n}" for n in NAMES["topics"]] + [f"format_{n}" for n in NAMES["formats"]]


def prompt(text, t):
    y = TAX[t]; choices = "\n".join(f"{LET[i]}: {c}" for i, c in enumerate(y["choices"]))
    body = y["template"].replace("URL: `{url}`\n", "").replace("{text}", text).replace("{choices}", choices)
    body = body.replace("The URL might help you understand the content. Avoid shortcuts such as word overlap between the page and the topic descriptions or simple patterns in the URL.",
                        "Avoid shortcuts such as word overlap between the page and the descriptions.")
    body = body.replace("The URL might help you understand the content. ", "")
    body = re.sub(r"Avoid shortcuts such as word overlap between the page and the (\w+) descriptions or simple patterns in the URL(, for example `[^`]*`[^.]*)?\.",
                  r"Avoid shortcuts such as word overlap between the page and the \1 descriptions.", body)
    body = re.sub(r"Start your response with the single-letter ID of the correct \w+ followed by an explanation\.", "", body)
    return "The web page may be in any language; judge it in its own language.\n\n" + body + "\nAnswer with only the letter."


def label(llm, texts, max_tokens=1500):
    from vllm import SamplingParams
    tk = llm.get_tokenizer()
    texts = [tk.decode(ids[:max_tokens]) if len(ids) > max_tokens else t for t, ids in zip(texts, tk(texts, add_special_tokens=False)["input_ids"])]
    msgs = [[{"role": "user", "content": prompt(x, t)}] for x in texts for t in ("topics", "formats")]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    res = []
    for i in range(len(texts)):
        row = {}
        for o, t, pre in zip(outs[2 * i:2 * i + 2], ("topics", "formats"), ("topic_", "format_")):
            p = [0.0] * 24
            for v in (o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}).values():
                s = v.decoded_token.strip().upper()
                if len(s) == 1 and s in LET:
                    p[LET.index(s)] += math.exp(v.logprob)
            z = sum(p)
            row.update({pre + n: (p[j] / z if z > 0 else float("nan")) for j, n in enumerate(NAMES[t])})
        res.append(row)
    return res


if __name__ == "__main__":
    import importlib.util
    import pandas as pd
    from vllm import LLM
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(HERE, "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    kw = {"quantization": os.environ["QUANT"]} if os.environ.get("QUANT") else {}
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=256,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    df = df.iloc[sh::ns].copy().reset_index(drop=True); C = int(os.environ.get("CHUNK", 2000)); parts = []
    for c in range(0, len(df), C):  # save after every chunk: a job that hits its time limit keeps what it finished
        part = df.iloc[c:c + C].reset_index(drop=True); parts.append(pd.concat([part, pd.DataFrame(label(llm, part.text.tolist()))], axis=1))
        pd.concat(parts, ignore_index=True).to_parquet(os.environ["LABEL_OUT"]); print("labelled", c + len(part), "of", len(df), flush=True)
    _m.hard_exit(0)
