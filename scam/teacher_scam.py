"""Qwen3.8-27B as a spam / scam teacher (vLLM venv). Classes: legitimate, spam (unwanted marketing or bulk messages),
fraud (phishing, scams, malware lures, social engineering). Scores = first-answer-token probabilities, renormalised.
Eval:  python scam/teacher_scam.py OUT.json EVAL_DIR     (unsafe score = p_spam + p_fraud)
Label: LABEL_IN=texts.parquet LABEL_OUT=out.parquet [SHARD=i NSHARD=n] python scam/teacher_scam.py - -
"""
import json, math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import load_evals, score, summarize

C = ["legitimate", "spam", "fraud"]
PROMPT = ("Classify the following message (an email, SMS, chat or social media message).\n"
          "- legitimate: a normal personal, work, transactional or service message, including genuine notifications from "
          "companies (order updates, real one-time codes, appointment reminders) and newsletters the recipient likely signed up for\n"
          "- spam: unsolicited bulk advertising or marketing, promotions, adult or dating ads, chain messages - unwanted, but not "
          "trying to steal money, credentials or data\n"
          "- fraud: phishing (fake login, bank, delivery or account-problem messages with links), scams (prizes, lotteries, "
          "inheritance, investment or crypto schemes, fake jobs, romance or family-emergency scams, advance-fee requests), "
          "requests for passwords, codes or payment details, or malware lures\n\n"
          "Message:\n<<<\n{text}\n>>>\n\nAnswer with one word: legitimate, spam or fraud.")


def probs(llm, texts):
    from vllm import SamplingParams
    msgs = [[{"role": "user", "content": PROMPT.format(text=t[:6000])}] for t in texts]
    outs = llm.chat(msgs, SamplingParams(temperature=0, max_tokens=1, logprobs=20), chat_template_kwargs={"enable_thinking": False})
    res = []
    for o in outs:
        lp = o.outputs[0].logprobs[0] if o.outputs[0].logprobs else {}
        p = [0.0, 0.0, 0.0]
        for v in lp.values():
            t = v.decoded_token.strip().lower()
            for i, c in enumerate(C):
                if len(t) >= 2 and c.startswith(t):
                    p[i] += math.exp(v.logprob)
        s = sum(p); res.append([x / s for x in p] if s > 0 else [1 / 3] * 3)
    return res


if __name__ == "__main__":
    import pandas as pd
    from vllm import LLM
    import importlib.util
    _sp = importlib.util.spec_from_file_location("sent_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=256,
              limit_mm_per_prompt={"image": 0, "video": 0})
    if os.environ.get("LABEL_IN"):
        df = pd.read_parquet(os.environ["LABEL_IN"]); sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns].copy(); p = probs(llm, df.text.tolist())
        df["p_legit"], df["p_spam"], df["p_fraud"] = zip(*p)
        df.to_parquet(os.environ["LABEL_OUT"]); print("labelled", len(df), flush=True); hard_exit(0)
    out, res = sys.argv[1], {}
    for name, df in load_evals(sys.argv[2]).items():
        p = probs(llm, df.text.tolist()); res[name] = score(df, [a[1] + a[2] for a in p])
        res[name]["fraud_share_pred"] = float(sum(a[2] > max(a[0], a[1]) for a in p) / len(p)); print(name, res[name], flush=True)
    res.update(summarize(res)); print({k: v for k, v in res.items() if k.startswith("_")}, flush=True)
    json.dump(res, open(out, "w"), indent=1); hard_exit(0)
