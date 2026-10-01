"""Qwen3-Reranker teacher scores with vLLM (vLLM venv), chunked with partial saves.
IN=mined.parquet PASSAGES=passages.parquet SHARD=i NSHARD=n OUT=file.parquet [MODEL=Qwen/Qwen3-Reranker-4B] [LIMIT=n]
python rerank/teacher_score_vllm.py -> rows + scores (log-odds log P(yes) - log P(no), same as teacher_score.py).
Prompt = the model card's template with the default web-search instruction; documents truncated to MAX_DOC_TOK tokens.
Runs as a 1-token generation with logprobs for "yes"/"no" and prefix caching (system prompt + query shared by 16 candidates).
"""
import math, os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))

PRE = ("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
       "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUF = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
INST = "Given a web search query, retrieve relevant passages that answer the query"

if __name__ == "__main__":
    from vllm import LLM, SamplingParams, TokensPrompt
    from common import hard_exit
    df = pd.read_parquet(os.environ["IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
    df = df.iloc[sh::ns].reset_index(drop=True)
    if os.environ.get("LIMIT"):
        df = df.iloc[: int(os.environ["LIMIT"])]
    ptext = dict(pd.read_parquet(os.environ["PASSAGES"])[["pid", "text"]].values)
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3-Reranker-4B"), max_model_len=1024, gpu_memory_utilization=0.9, max_num_seqs=512,
              enable_prefix_caching=True, logprobs_mode="processed_logprobs")   # logprobs after the yes/no restriction: both always present
    tok = llm.get_tokenizer(); yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre, suf = tok.encode(PRE, add_special_tokens=False), tok.encode(SUF, add_special_tokens=False)
    MD = int(os.environ.get("MAX_DOC_TOK", 400))
    sp = SamplingParams(temperature=0, max_tokens=1, logprobs=20, allowed_token_ids=[yes, no])
    out_path = os.environ["OUT"]; scores = [None] * len(df); CH = int(os.environ.get("CHUNK", 4000))
    for s in range(0, len(df), CH):
        rows = df.iloc[s:s + CH]; prompts, owner = [], []
        for ri, (q, cs) in enumerate(zip(rows["query"], rows.cands)):
            qi = tok.encode(f"<Instruct>: {INST}\n<Query>: {q}\n<Document>: ", add_special_tokens=False)
            for c in cs:
                prompts.append(TokensPrompt(prompt_token_ids=pre + qi + tok.encode(ptext[c], add_special_tokens=False)[:MD] + suf)); owner.append(ri)
        outs = llm.generate(prompts, sp, use_tqdm=False)
        vals = []
        for o in outs:
            lp = o.outputs[0].logprobs[0]
            ly = lp[yes].logprob if yes in lp else None; ln = lp[no].logprob if no in lp else None
            if ly is None or ln is None or not (math.isfinite(ly) and math.isfinite(ln)):   # should not happen with processed logprobs
                ly, ln = (0.0, -20.0) if ln is None or not math.isfinite(ln) else (-20.0, 0.0)
            vals.append(ly - ln)
        k = 0
        for ri, cs in enumerate(rows.cands):
            scores[s + ri] = vals[k:k + len(cs)]; k += len(cs)
        done = s + len(rows)
        part = df.iloc[:done].copy(); part["scores"] = scores[:done]; part.to_parquet(out_path)   # partial save after every chunk
        print("scored queries", done, "of", len(df), flush=True)
    print("done", flush=True)
    hard_exit(0)
