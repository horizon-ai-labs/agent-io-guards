"""Teacher compressions for a multilingual LLMLingua-2-style prompt compressor. Qwen3.8-27B (Apache-2.0) deletes words from a text
(no reordering, no new words), following the LLMLingua-2 data-distillation instruction (Pan et al. 2024), adapted so the model keeps
the text's language. Greedy decoding. Labels (kept / dropped per word) are derived later by alignment (compress/align.py).
IN=<parquet with id,text,lang> SHARD=i NSHARD=n OUT=<file.jsonl> [VARIANT=target] [MODEL=...] python compress/teacher_compress.py   (vLLM venv)
Output rows {id, lang, out} appended after every chunk of 2,000 inputs (restartable)."""
import json, os, sys
import pandas as pd
import importlib.util
_sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit   # kills the vLLM engine child too

INSTR = """Compress the given text to short expressions, such that you can reconstruct it as close as possible to the original. Unlike the usual text compression, you must comply with the 5 conditions below:
1. You can ONLY remove unimportant words.
2. Do not reorder the original words.
3. Do not change the original words (no translation, no spelling or case changes).
4. Do not use abbreviations or emojis.
5. Do not add new words or symbols.
Compress the original aggressively by removing words only. Compress it as short as you can, while retaining as much information as possible.{target} The text can be in any language; keep its language.
Reply with the compressed text only.

Text to compress:
{text}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    shard, nshard, out = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1)), os.environ["OUT"]
    d = pd.read_parquet(os.environ["IN"]); d = d[d.id % nshard == shard]
    done = set()
    if os.path.exists(out):
        done = {json.loads(l)["id"] for l in open(out)}
    d = d[~d.id.isin(done)].reset_index(drop=True)
    print("todo", len(d), "done", len(done), flush=True)
    if len(d):
        llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512,
                  limit_mm_per_prompt={"image": 0, "video": 0})
        tok = llm.get_tokenizer()
        ntok = [len(tok(t).input_ids) for t in d.text]
        d = d[[n <= 3000 for n in ntok]].reset_index(drop=True)   # very token-dense passages (URLs, tables, base64) are skipped
        print("skipped (> 3000 tokens):", sum(n > 3000 for n in ntok), flush=True)
        for s in range(0, len(d), 2000):
            part = d.iloc[s:s + 2000]
            tgt = {"": "", "target": " Keep only about 40% of the words."}[os.environ.get("VARIANT", "")]
            msgs = [[{"role": "user", "content": INSTR.replace("{target}", tgt).replace("{text}", t)}] for t in part.text]   # not .format: texts contain braces
            mx = [min(2048, int(len(tok(t).input_ids) * 1.1) + 16) for t in part.text]
            outs = llm.chat(msgs, [SamplingParams(temperature=0.0, max_tokens=m) for m in mx], chat_template_kwargs={"enable_thinking": False},
                            use_tqdm=False)
            with open(out, "a") as f:
                for (i, l), o in zip(zip(part.id, part.lang), outs):
                    f.write(json.dumps(dict(id=int(i), lang=l, out=o.outputs[0].text, finish=o.outputs[0].finish_reason), ensure_ascii=False) + "\n")
            print("chunk", s, "->", s + len(part), flush=True)
    hard_exit(0)
