"""results.json for the content-safety leaderboard Space.  python release/safety_leaderboard/build.py"""
import json

SETS = [("PolyGuard prompts", "polyguard_prompt", "f1", "17 languages"), ("PolyGuard responses", "polyguard_response", "f1", "17 languages"),
        ("BeaverTails (unseen)", "beavertails_unseen", "f1", "responses, en"), ("ToxicChat", "toxicchat", "f1", "real prompts, en"),
        ("OpenAI moderation", "openai_mod", "f1", "en"), ("XSTest", "xstest", "f1", "over-blocking, en"),
        ("textdetox", "textdetox", "f1", "toxicity, 14 languages"), ("Aya red-teaming", "aya_redteaming", "recall", "recall, 8 languages"),
        ("SimpleSafetyTests", "simplesafety", "recall", "recall, en")]
bl = json.load(open("release/evals/safety_baselines.json"))
ours = {k: list(json.load(open(f"release/evals/safety_{k}.json")).values())[0] for k in ["base", "small"]}
MODELS = [("Horizon-Labs/content-safety-guard-base", "", ours["base"], "308M", "apache-2.0"),
          ("Horizon-Labs/content-safety-guard-small", "", ours["small"], "141M", "apache-2.0"),
          ("Qwen/Qwen3Guard-Gen-0.6B", "strict", bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_strict"], "0.6B (generative)", "apache-2.0"),
          ("Qwen/Qwen3Guard-Gen-0.6B", "loose", bl["qwen3guard:Qwen/Qwen3Guard-Gen-0.6B_loose"], "0.6B (generative)", "apache-2.0"),
          ("Qwen/Qwen3Guard-Gen-8B", "strict", bl["qwen3guard:Qwen/Qwen3Guard-Gen-8B_strict"], "8B (generative)", "apache-2.0"),
          ("Qwen/Qwen3Guard-Gen-8B", "loose", bl["qwen3guard:Qwen/Qwen3Guard-Gen-8B_loose"], "8B (generative)", "apache-2.0"),
          ("llm-semantic-router/Vela-1.0-Encoder-307M-Shield", "", bl["llm-semantic-router/Vela-1.0-Encoder-307M-Shield"], "307M", "apache-2.0"),
          ("ibm-granite/granite-guardian-hap-125m", "", bl["ibm-granite/granite-guardian-hap-125m"], "125M", "apache-2.0"),
          ("unitary/unbiased-toxic-roberta", "", bl["unitary/unbiased-toxic-roberta"], "125M", "apache-2.0"),
          ("unitary/multilingual-toxic-xlm-roberta", "", bl["unitary/multilingual-toxic-xlm-roberta"], "278M", "apache-2.0")]
rows = []
for mid, mode, e, size, lic in MODELS:
    r = dict(model=mid, mode=mode, size=size, license=lic)
    for name, key, m, _ in SETS:
        r[name] = e[key][m]
        if m == "f1":
            r[name + " FPR"] = e[key]["fpr"]
    f1s = [r[n] for n, k, m, _ in SETS if m == "f1"]
    r["macro"] = sum(f1s) / len(f1s)
    rows.append(r)
json.dump(dict(sets=[(n, g, m) for n, k, m, g in SETS], rows=rows, snapshot="2026-09-27"), open("release/safety_leaderboard/results.json", "w"), indent=1)
for r in sorted(rows, key=lambda r: -r["macro"]):
    print(f"{r['model'][:45]:45s} {r['mode']:6s} macro {r['macro']:.3f}")
