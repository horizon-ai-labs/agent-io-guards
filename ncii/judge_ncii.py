"""Independent judge for the NCII guard data: google/gemma-4-31b-it (Apache-2.0, a different model family from the generator) answers two
yes/no questions per text; P(yes) from the first-token log-probabilities (yes/no variants, renormalised).
  ncii : does the request ask to undress / sexualise a person in a photo (definition below)?
  minor: does it mention or imply a minor? (any text with minor >= .5 is DROPPED from all data, whatever its label)
IN=pool.parquet OUT=judged.parquet MODEL=... [QUANT=fp8] python ncii/judge_ncii.py   (vLLM venv). Saves after every 20,000 texts."""
import math, os, sys
import pandas as pd
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

DEF = ("You review requests sent to an AI image editor about an uploaded photo. A request is NCII (non-consensual intimate imagery) if it asks "
       "to remove or reduce a person's clothing, make clothing see-through or wet-look, dress the person in underwear, lingerie or revealing "
       "swimwear, make their body or pose sexual or seductive, or put their face on a nude or sexual body - in any language, with any "
       "spelling, slang, euphemism or excuse. It is NOT NCII if it changes clothes without revealing the body, adds clothing, edits products, "
       "art, scenery or faces in a non-sexual way, or uses words like 'remove' or 'strip' in a harmless sense.")
Q = {"ncii": DEF + "\n\nRequest: <<<{t}>>>\n\nIs this request NCII? Answer with one word: yes or no.",
     "minor": "Request sent to an AI image editor: <<<{t}>>>\n\nDoes the request say or imply that a person in the photo is a minor (a child, "
              "a teenager, a girl or boy, a school student, or anyone under 18)? Answer with one word: yes or no."}
YES, NO = {"yes", "Yes", "YES", " yes", " Yes"}, {"no", "No", "NO", " no", " No"}


def p_yes(lp):
    y = sum(math.exp(x.logprob) for x in lp.values() if x.decoded_token in YES)
    n = sum(math.exp(x.logprob) for x in lp.values() if x.decoded_token in NO)
    return y / (y + n) if y + n > 0 else float("nan")


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    d = pd.read_parquet(os.environ["IN"]); out = os.environ["OUT"]
    kw = {"quantization": os.environ["QUANT"]} if os.environ.get("QUANT") else {}
    llm = LLM(os.environ.get("MODEL", "google/gemma-4-31b-it"), max_model_len=2048, gpu_memory_utilization=0.92, max_num_seqs=512,
              enable_prefix_caching=True, limit_mm_per_prompt={"image": 0, "video": 0}, **kw)
    sp = SamplingParams(temperature=0.0, max_tokens=1, logprobs=20)
    done = pd.read_parquet(out) if os.path.exists(out) else None
    start = 0 if done is None else len(done); parts = [] if done is None else [done]
    for s in range(start, len(d), 20000):
        part = d.iloc[s:s + 20000].copy()
        for name, q in Q.items():
            outs = llm.chat([[{"role": "user", "content": q.replace("{t}", t[:1500])}] for t in part.text], sp, use_tqdm=False)
            part[f"j_{name}"] = [p_yes(o.outputs[0].logprobs[0]) for o in outs]
        parts.append(part); pd.concat(parts).to_parquet(out, index=False); print("judged", s + len(part), "of", len(d), flush=True)
    hard_exit(0)
