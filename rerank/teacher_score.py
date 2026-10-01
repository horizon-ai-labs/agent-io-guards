"""Teacher relevance scores for (query, candidate) pairs.   (HF transformers, GPU)
TEACHER=qwen3rr:Qwen/Qwen3-Reranker-4B | <cross-encoder id>  IN=mined.parquet PASSAGES=passages.parquet SHARD=i NSHARD=n
OUT=file.parquet python rerank/teacher_score.py -> rows of IN (sharded) + scores (list of floats aligned with cands;
Qwen3-Reranker: log-odds log P(yes) - log P(no); cross-encoder: raw logit).
"""
import os, sys
import numpy as np, pandas as pd, torch

T = os.environ.get("TEACHER", "qwen3rr:Qwen/Qwen3-Reranker-4B"); BS = int(os.environ.get("BS", 64)); ML = int(os.environ.get("MAX_LEN", 512))
df = pd.read_parquet(os.environ["IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
df = df.iloc[sh::ns].reset_index(drop=True)
ptext = dict(pd.read_parquet(os.environ["PASSAGES"])[["pid", "text"]].values)
pairs = [(q, ptext[c]) for q, cs in zip(df["query"], df.cands) for c in cs]
print("pairs", len(pairs), flush=True)
out = np.zeros(len(pairs), dtype=np.float32); order = np.argsort([len(q) + len(d) for q, d in pairs])
if T.startswith("qwen3rr:"):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T[8:]); m = AutoModelForCausalLM.from_pretrained(T[8:], torch_dtype=torch.bfloat16).cuda().eval()
    PRE = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct provided. "
                     "Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    SUF = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    INST = "Given a web search query, retrieve relevant passages that answer the query"
    with torch.no_grad():
        for s in range(0, len(pairs), BS):
            b = order[s:s + BS]
            ids = [PRE + tok.encode(f"<Instruct>: {INST}\n<Query>: {pairs[i][0]}\n<Document>: {pairs[i][1]}", add_special_tokens=False)[: ML - len(PRE) - len(SUF)] + SUF for i in b]
            L = max(map(len, ids)); inp = torch.full((len(ids), L), tok.pad_token_id); att = torch.zeros((len(ids), L), dtype=torch.long)
            for j, x in enumerate(ids):
                inp[j, L - len(x):] = torch.tensor(x); att[j, L - len(x):] = 1
            lo = m(input_ids=inp.cuda(), attention_mask=att.cuda(), logits_to_keep=1).logits[:, -1, :].float()
            out[b] = (lo[:, yes] - lo[:, no]).cpu().numpy()
            if s % (BS * 500) == 0:
                print(s, flush=True)
else:
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(T); m = AutoModelForSequenceClassification.from_pretrained(T, torch_dtype=torch.bfloat16).cuda().eval()
    with torch.no_grad():
        for s in range(0, len(pairs), BS):
            b = order[s:s + BS]
            enc = tok([pairs[i][0] for i in b], [pairs[i][1] for i in b], truncation="only_second", max_length=ML, padding=True, return_tensors="pt")
            lo = m(**{k: v.cuda() for k, v in enc.items()}).logits.float()
            out[b] = (lo[:, 0] if lo.shape[1] == 1 else lo[:, 1] - lo[:, 0]).cpu().numpy()
k = 0; sc = []
for cs in df.cands:
    sc.append(out[k:k + len(cs)].tolist()); k += len(cs)
df["scores"] = sc; df.to_parquet(os.environ["OUT"]); print("scored", len(df), flush=True)
