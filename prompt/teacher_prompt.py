"""LLM labeller for the prompt task & complexity classifier (vLLM venv). The user prompt comes first (shared cached prefix), then one
question per output:
- task: which of 11 task types (NVIDIA prompt-task taxonomy) -> probability per type from the answer-letter distribution;
- creativity, reasoning, contextual_knowledge, domain_knowledge, constraints, complexity: 0-9 rating -> expected value / 9 (0-1).
LABEL_IN=in.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] [MODEL=... QUANT=fp8] python prompt/teacher_prompt.py
(the same script, with MODEL=google/gemma-4-31b-it QUANT=fp8, produces the independent judge labels of the test set)"""
import math, os, sys

TASKS = ["open_qa", "closed_qa", "summarization", "text_generation", "code_generation", "chatbot", "classification", "rewrite",
         "brainstorming", "extraction", "other"]
TASK_DESC = ["Open QA: a question answered from general knowledge", "Closed QA: a question answered from text given in the prompt",
             "Summarization: summarise given text", "Text generation: write new text (stories, essays, emails, poems, articles)",
             "Code generation: write, fix or explain code", "Chatbot: casual conversation, role-play or chit-chat",
             "Classification: assign a label or category to given text", "Rewrite: translate, paraphrase, edit or restyle given text",
             "Brainstorming: generate ideas, options or lists of suggestions", "Extraction: pull specific information out of given text",
             "Other: none of the above (e.g. maths problems, unclear requests)"]
L = "ABCDEFGHIJK"
DIMS = {"creativity": "How much creativity is needed to respond well?",
        "reasoning": "How much logical reasoning or multi-step thinking is needed to respond well?",
        "contextual_knowledge": "How much background information that is NOT in the prompt is needed to respond well?",
        "domain_knowledge": "How much specialised knowledge or expertise in a subject area is needed to respond well?",
        "constraints": "How many explicit constraints or conditions (format, length, style, rules) does the prompt impose?",
        "complexity": "Overall, how complex and demanding is this prompt for an AI assistant?"}
LABELS = [f"task_{t}" for t in TASKS] + list(DIMS)


def head(text):
    return f"Here is a user prompt sent to an AI assistant (it may be in any language):\n<<<\n{text}\n>>>\n\n"


def q_task(text):
    opts = "\n".join(f"{L[i]}. {d}" for i, d in enumerate(TASK_DESC))
    return head(text) + f"Which task type is this prompt?\n{opts}\nAnswer with one letter."


def q_dim(text, d):
    return head(text) + f"{DIMS[d]} Rate from 0 (none at all) to 9 (extremely high). Answer with one digit."


def label(llm, texts, max_tokens=1500):
    from vllm import SamplingParams
    tk = llm.get_tokenizer()
    texts = [tk.decode(ids[:max_tokens]) if len(ids) > max_tokens else t for t, ids in zip(texts, tk(texts, add_special_tokens=False)["input_ids"])]
    msgs = []
    for t in texts:
        msgs.append([{"role": "user", "content": q_task(t)}]); msgs += [[{"role": "user", "content": q_dim(t, d)}] for d in DIMS]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    k = 1 + len(DIMS); res = []
    for i in range(len(texts)):
        o = outs[i * k:(i + 1) * k]; row = {}
        p = [0.0] * len(L)
        for v in (o[0].outputs[0].logprobs[0] if o[0].outputs[0].logprobs else {}).values():
            t = v.decoded_token.strip().upper()
            if len(t) == 1 and t in L:
                p[L.index(t)] += math.exp(v.logprob)
        z = sum(p) or 1.0; row.update({f"task_{TASKS[j]}": p[j] / z for j in range(len(L))})
        for d, oo in zip(DIMS, o[1:]):
            q = [0.0] * 10
            for v in (oo.outputs[0].logprobs[0] if oo.outputs[0].logprobs else {}).values():
                t = v.decoded_token.strip()
                if len(t) == 1 and t in "0123456789":
                    q[int(t)] += math.exp(v.logprob)
            zq = sum(q); row[d] = sum(j * x for j, x in enumerate(q)) / zq / 9 if zq > 0 else float("nan")
        res.append(row)
    return res


if __name__ == "__main__":
    import importlib.util
    import pandas as pd
    from vllm import LLM
    here = os.path.dirname(os.path.abspath(__file__))
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(here, "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m)
    kw = {"quantization": os.environ["QUANT"]} if os.environ.get("QUANT") else {}
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    df = df.iloc[sh::ns].copy().reset_index(drop=True)
    df = pd.concat([df, pd.DataFrame(label(llm, df.text.tolist()))], axis=1)
    df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True)
    _m.hard_exit(0)
