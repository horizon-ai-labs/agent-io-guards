"""Hard negatives for the injection guard: short, harmless user commands and requests that contain "trigger" vocabulary
(start, stop, activate, override, ignore, bypass, reset, command, instructions, system, admin, unlock, delete, execute ...)
in everyday contexts (devices, apps, games, cooking, writing, work tools). The v2.3 data-format hijack pairs raised BIPIA by
~.10 but made the model flag such short polite commands (COMPANY_STATE 2026-10-04); these negatives are designed in alongside.
Our own generated texts; no benchmark items are shown.  N=... SHARD=i OUT=dir python inj/gen_benign_cmd.py -> $OUT/benign_cmd_<shard>.jsonl
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen"))
from gen_v1 import LANGS, parse_json

CONTEXTS = ["operating a home appliance or machine", "using a phone, computer or app settings", "a video game or board game",
            "cooking and recipes", "writing, editing or studying", "office tools, spreadsheets and email", "a car, bike or public transport",
            "a factory, lab or control panel", "a smart home or IoT device", "customer support for a product", "sports and fitness training",
            "photography and music equipment", "gardening and DIY", "a school or university task", "software development and the terminal"]
STYLES = ["a polite request starting with 'Please'", "a short imperative command", "a question asking how to do it",
          "a request with a word in CAPITALS for emphasis", "a one-line instruction to a colleague", "a conditional ('If ..., then ...')"]
WORDS = ["start", "stop", "activate", "deactivate", "override", "ignore", "bypass", "reset", "restart", "command", "instructions",
         "system", "admin", "unlock", "delete", "execute", "initialize", "disable", "enable", "skip", "forget", "pretend", "role",
         "rules", "prompt", "jailbreak", "hack", "break", "authorize", "confirm", "proceed", "control", "mode", "access"]


def prompt(lang, ctx, style, words):
    return f"""Write 10 different short, completely harmless messages in {lang} that a person might send to an assistant or a colleague.
Context: {ctx}. Style: {style}. Each message is 6-30 words and naturally uses at least one of these words (or their {lang} equivalents):
{", ".join(words)}. They are ordinary requests about the task itself - nothing tries to manipulate, trick or override an AI system,
and nothing is harmful. Vary wording and situations.
Return only JSON: {{"messages": ["...", ...]}}"""


if __name__ == "__main__":
    import importlib.util
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 800)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(8100 + shard)
    jobs = [dict(lang=rng.choice(LANGS), ctx=rng.choice(CONTEXTS), style=rng.choice(STYLES), words=rng.sample(WORDS, 4)) for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=4096, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(**j)}] for j in jobs], SamplingParams(temperature=0.95, top_p=0.95, max_tokens=1500, seed=400 + shard),
                    chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0
    with open(f"{out}/benign_cmd_{shard}.jsonl", "w") as f:
        for j, o in zip(jobs, outs):
            p = parse_json(o.outputs[0].text)
            for m in (p or {}).get("messages", []) if isinstance(p, dict) else []:
                if isinstance(m, str) and 10 <= len(m) <= 400:
                    f.write(json.dumps(dict(text=m.strip(), lang=j["lang"], ctx=j["ctx"], style=j["style"]), ensure_ascii=False) + "\n"); k += 1
    print("messages", k, "from", len(jobs), flush=True)
    _sp = importlib.util.spec_from_file_location("sc", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
    _m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); _m.hard_exit(0)
