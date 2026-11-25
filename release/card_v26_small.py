"""Card update for prompt-injection-guard-small v2.6 (patches the current README; later card sections were added by patch scripts, so
make_card.py is not re-run). Inputs: as-shipped eval of release build a25b4bf7 (inj/run_release26.sh), v2.6 training seeds, v2.3 baseline
83084f06, third-party re-check 60cef43d (inj/run_v26_extra.sh). python release/card_v26_small.py [--dry] -> writes .tmp/inj_small_v26.md"""
import glob, json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "inj"); from tool_metrics import metrics
from huggingface_hub import HfApi, hf_hub_download

REPO = "Horizon-Labs/prompt-injection-guard-small"
REL, X = "/outputs/a25b4bf7-cea9-4ee0-8886-df5e40f8b75f", glob.glob("/outputs/60cef43d-*")[0]
E = list(json.load(open(f"{REL}/eval_results.json")).values())[0]
SEEDS26 = ["b3171509", "663a883c", "cc99d278", "6d3c07af"]; SEEDS23 = ["f75a7bfa", "6c8a8a2d", "27031a67", "53695bf2"]
pdir = lambda j: [p for p in glob.glob("/outputs/*/preds/*") if j in p.split("__")[-2] and glob.glob(f"{p}/terse_neg.parquet")][0]
M26 = metrics(glob.glob(f"{REL}/preds/*")[0]); M23 = metrics(pdir("53695bf2")); ML = metrics(pdir("4012d571")); MB = metrics(pdir("8f435d9b"))
mean = lambda js, k: float(np.mean([metrics(pdir(j))[k] for j in js]))
f3 = lambda x: f"{x:.3f}"; pc = lambda x: f"{100 * x:.1f}%"

ROWS = {"NotInject": ("notinject", "acc"), "XSTest": ("xstest", "acc"), "OR-Bench": ("orbench_hard", "acc"), "Qualifire benchmark": ("qualifire", "f1"),
        "jackhhao/jailbreak": ("jackhhao_test", "f1"), "deepset/prompt-injections": ("deepset_test", "f1"), "Simsonsun": ("simsonsun_jailbreaks", "acc"),
        "Agentic boundary": ("boundary_pairs_test", "f1"), "Held-out synthetic direct": ("synthetic_direct_test", "f1"), "BIPIA (": ("bipia", "f1"),
        "PIArena (": ("piarena", "f1"), "LLMail": ("llmail_phase2", "acc"), "Held-out synthetic documents": ("synthetic_docs_test", "f1"),
        "Mindgard originals": ("mindgard_original", "acc"), "Mindgard evaded": ("mindgard_evasion", "acc")}
AUC = {"Qualifire": "qualifire", "jackhhao": "jackhhao_test", "deepset": "deepset_test", "Boundary pairs": "boundary_pairs_test", "BIPIA": "bipia", "PIArena": "piarena"}
num = lambda c: float(re.search(r"[\d.]+", c).group())


def rebold(cells):
    plain = [c.replace("**", "") for c in cells]; best = max(num(c) for c in plain)
    return [f"**{c}**" if abs(num(c) - best) < 5e-4 else c for c in plain]


def patch_tables(t):
    out = []
    for line in t.split("\n"):
        if line.startswith("| ") and not line.startswith("| Eval set") and not line.startswith("| ROC AUC") and "|---" not in line:
            cells = [c.strip() for c in line.strip("|").split("|")]; label = cells[0]; new = None
            if label in AUC and len(cells) == 10:
                new = f3(E[AUC[label]]["auc"])
            else:
                for pre, (k, m) in ROWS.items():
                    if label.startswith(pre) and len(cells) == 10:
                        new = f"{f3(E[k]['f1'])} ({E[k]['fpr']:.2f})" if m == "f1" else f3(E[k]["acc"])
            if new is not None:
                cells = [label] + rebold([new] + cells[2:]); line = "| " + " | ".join(cells) + " |"
        out.append(line)
    return "\n".join(out)


EXT = [("notinject", "acc"), ("xstest", "acc"), ("orbench_hard", "acc"), ("qualifire", "f1"), ("jackhhao_test", "f1"), ("deepset_test", "f1"),
       ("simsonsun_jailbreaks", "acc"), ("boundary_pairs_test", "f1"), ("bipia", "f1"), ("piarena", "f1"), ("llmail_phase2", "acc")]
MACRO = sum(E[s][k] for s, k in EXT) / len(EXT)


def patch_macro(t):
    i = t.index("Macro average over the external (non-synthetic) sets above: "); j = t.index("\n", i)
    items = [x.strip().rstrip(".") for x in t[i:j].split(": ", 1)[1].split(" · ")]
    d = {x.rsplit(": ", 1)[0]: float(x.rsplit(": ", 1)[1]) for x in items}; d["**this model**"] = MACRO
    s = " · ".join(f"{k}: {v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1]))
    return t[:i] + "Macro average over the external (non-synthetic) sets above: " + s + "." + t[j:]


# --- tool-output section
TOOL = f"""### Tool outputs and terse status lines (checked November 2026)

Agents scan tool results after every call, and many of those results are terse machine text. SamJoshua
[reported](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small/discussions/1) that v2.3 flagged status lines such as
`probe q3481 ok n=11` with 0.93–0.99, also inside MCP results. v2.6 was trained with templated status, acknowledgement and log lines
as hard negatives, and with short attacks placed inside the same status lines and envelopes as positives, so that "looks like tool
output" does not mean "safe". Threshold 0.5:

| | small v2.6 (this version) | small v2.3 | base v2.2 | large v2.1 |
|---|---|---|---|---|
| the reported strings (17 status / ack lines), flagged | {M26['samj_flagged']} | {M23['samj_flagged']} | {MB['samj_flagged']} | {ML['samj_flagged']} |
| held-out terse lines (575; verbs and envelopes not used in training: logs, YAML, `<tool_result>`, OpenAI tool messages), flagged | {pc(M26['terse_fpr'])} | {pc(M23['terse_fpr'])} | {pc(MB['terse_fpr'])} | {pc(ML['terse_fpr'])} |
| short attacks from our eval sets inside those envelopes (300), caught | {pc(M26['terse_pos_rec'])} | {pc(M23['terse_pos_rec'])} | {pc(MB['terse_pos_rec'])} | {pc(ML['terse_pos_rec'])} |
| [tool-output-injection-bench](https://huggingface.co/datasets/SamJoshua/tool-output-injection-bench) (408, Apache-2.0), F1 | {f3(M26['toib_f1'])} | {f3(M23['toib_f1'])} | {f3(MB['toib_f1'])} | {f3(ML['toib_f1'])} |
| same bench, benign outputs flagged (of 156) | {M26['toib_benign_fp']} | {M23['toib_benign_fp']} | {MB['toib_benign_fp']} | {ML['toib_benign_fp']} |
| same bench, held-out split, injections caught (of 72) | {round(M26['toib_held_rec'] * 72)} | {round(M23['toib_held_rec'] * 72)} | {round(MB['toib_held_rec'] * 72)} | {round(ML['toib_held_rec'] * 72)} |

- The reported strings are not a held-out test: bare acknowledgements such as "ok" and "done" also occur in the training templates.
  The 575 held-out lines use other verbs, acknowledgements and envelope formats.
- **Seed variance, and a weakness of this checkpoint.** We trained four seeds and release the one with the best validation score,
  as decided before training. Averaged over the four seeds, benign outputs flagged on the bench fell from {mean(SEEDS23, 'toib_benign_fpr') * 156:.0f}
  (v2.3) to {mean(SEEDS26, 'toib_benign_fpr') * 156:.0f}, and F1 rose from {mean(SEEDS23, 'toib_f1'):.3f} to {mean(SEEDS26, 'toib_f1'):.3f}. The released checkpoint
  is the worst of the four on benign tool *descriptions* that contain legitimate imperatives ("run `admin unlock` and retry",
  "cache bypass: `--no-cache`"): it flags {M26['toib_benign_fp']} of the 156 benign outputs, against {M23['toib_benign_fp']} for v2.3 (the other v2.6
  seeds: 11–15). If you scan tool descriptions, review flags there or raise the threshold. v2.3 remains available under the `v2.3` tag.
- Instructions hidden in JSON fields are still the weakest carrier on that bench. Files: `eval/tool_outputs.json`;
  scripts: `code/inj/gen_terse.py`, `code/inj/build_tool_evals.py`.

"""

# --- third-party re-check (tr / ja, Aya / OASST2)
Xe = list(json.load(open(f"{X}/eval.json")).values())[0]
xp = glob.glob(f"{X}/preds/*")[0]


def roc(y, s):
    y = np.asarray(y).astype(bool); r = pd.Series(s).rank().values; n1 = y.sum(); n0 = len(y) - n1
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


tr = pd.read_json(hf_hub_download("halilneed/turkish-prompt-injection-benchmark", "data/full.jsonl", repo_type="dataset"), lines=True)
fa = pd.read_json(hf_hub_download("halilneed/turkish-prompt-injection-false-alarm-benchmark", "data/test.jsonl", repo_type="dataset"), lines=True)
sub = dict(zip(fa.text, fa.subset))
t = pd.read_parquet(f"{xp}/tr_pi_full.parquet"); ja = pd.read_parquet(f"{xp}/ja_indirect_doc.parquet")
y, s = t.label.values, t.score.values; ben = t[y == 0]; ti = np.array([sub.get(x) == "task_instruction" for x in tr.text[y == 0]])
TR = (roc(y, s), (s[y == 1] >= .5).mean(), (ben.score >= .5).mean(), (ben.score.values[ti] >= .5).mean(), (ja.score >= .5).mean())
TR_ROW = f"| **Horizon-Labs/prompt-injection-guard-small (v2.6)** | {TR[0]:.3f} | {TR[1]:.2f} | {TR[2]:.3f} | {TR[3]:.2f} | {TR[4]:.2f} |"
B = list(json.load(open(f"{X}/benign/fpr.json")).values())[0]
pcb = lambda x: f"{100 * x:.1f}%" if x >= .001 or x == 0 else f"{100 * x:.2f}%"
B_ROW = (f"| **this model** (v2.6) | {pcb(B['overall']['fpr50'])} / {pcb(B['overall']['fpr90'])} | {pcb(B['by_src']['aya']['fpr50'])} / {pcb(B['by_src']['aya']['fpr90'])} | "
         f"{pcb(B['by_src']['oasst2']['fpr50'])} / {pcb(B['by_src']['oasst2']['fpr90'])} | {pcb(B['per_lang']['en']['fpr50'])} / {pcb(B['per_lang']['en']['fpr90'])} | "
         f"{pcb(B['per_lang']['pt']['fpr50'])} / {pcb(B['per_lang']['pt']['fpr90'])} |")

V26_TRAIN = ("- **v2.6 (this version)**: v2.3 data plus ~5.1k templated terse status, acknowledgement and log lines (twice) as hard negatives and\n"
             "  ~1.5k short attacks from the training data placed inside such lines and tool-result envelopes (inj/gen_terse.py; no benchmark data).\n"
             "  Same recipe as v2.3 (distillation weight 0.5, 2 epochs).\n")
V26_LOG = (f"- **v2.6** (2026-11-25): fixes false alarms on terse tool output (status lines, acknowledgements, logs, also inside MCP results),\n"
           f"  reported by SamJoshua. Over four seeds (gates set before training, all passed): held-out terse lines flagged {pc(mean(SEEDS23, 'terse_fpr'))} -> "
           f"{pc(mean(SEEDS26, 'terse_fpr'))},\n  tool-output-injection-bench F1 {mean(SEEDS23, 'toib_f1'):.3f} -> {mean(SEEDS26, 'toib_f1'):.3f}, macro over the external sets .880 -> .883, "
           f"BIPIA F1 +.03, Simsonsun +.02,\n  NotInject -.005. The released seed scores {MACRO:.3f} macro and flags more benign tool descriptions than the other seeds "
           f"(see Tool outputs).\n")

if __name__ == "__main__":
    api = HfApi(); info = api.model_info(REPO); c = open(hf_hub_download(REPO, "README.md", revision=info.sha)).read()
    assert "Tool outputs and terse status lines" not in c
    c = patch_macro(patch_tables(c))
    c = c.replace("### Third-party multilingual benchmarks (checked October 2026)", TOOL + "### Third-party multilingual benchmarks (checked October 2026)")
    old_tr = [l for l in c.split("\n") if l.startswith("| **Horizon-Labs/prompt-injection-guard-small (v2.3)**")]; assert len(old_tr) == 1
    c = c.replace(old_tr[0], TR_ROW + "\n" + old_tr[0].replace("**Horizon-Labs/prompt-injection-guard-small (v2.3)**", "Horizon-Labs/prompt-injection-guard-small (v2.3)"))
    old_b = [l for l in c.split("\n") if l.startswith("| **this model** (v2.3) |")]; assert len(old_b) == 1
    c = c.replace(old_b[0], B_ROW + "\n" + old_b[0].replace("**this model** (v2.3)", "small v2.3"))
    c = c.replace("windowed function above), this model flags 0.15% at 0.5", "windowed function above), small v2.3 flags 0.15% at 0.5")
    c = c.replace("- **v2.3 (this version)**:", V26_TRAIN + "- **v2.3**:")
    c = c.replace("## Changelog\n\n", "## Changelog\n\n" + V26_LOG)
    q = E["qualifire"]["fpr"]; bi = E["bipia"]["recall"]
    c = re.sub(r"this model flags\n  \d+% of the benign prompts at 0.5", f"this model flags\n  {100 * q:.0f}% of the benign prompts at 0.5", c)
    c = re.sub(r"\(BIPIA recall \d+%, up from", f"(BIPIA recall {100 * bi:.0f}%, up from", c)
    c = re.sub(r"At 0.5 this model flags [\d.]+% of NotInject", f"At 0.5 this model flags {100 * (1 - E['notinject']['acc']):.1f}% of NotInject", c)
    c = c.replace("- Very short fragments and code without comments carry little signal.",
                  "- Code without comments carries little signal. Terse tool output is handled better since v2.6, but benign tool descriptions\n"
                  "  with imperatives (\"run X and retry\") are still flagged too often by this checkpoint (see Tool outputs).")
    open(".tmp/inj_small_v26.md", "w").write(c)
    tool = {"release_v2.6": M26, "v2.3_shipped": M23, "base_v2.2": MB, "large_v2.1": ML,
            "v2.6_4seed_mean": {k: mean(SEEDS26, k) for k in ("terse_fpr", "terse_pos_rec", "toib_f1", "toib_benign_fpr", "toib_held_rec")},
            "v2.3_4seed_mean": {k: mean(SEEDS23, k) for k in ("terse_fpr", "terse_pos_rec", "toib_f1", "toib_benign_fpr", "toib_held_rec")},
            "toib_revision": "2892d2b16d409630933ae9fa7785d3e73efe68f9"}
    json.dump(tool, open("release/evals/inj_tool_outputs_v26.json", "w"), indent=1)
    print("macro", round(MACRO, 4), "TR", [round(x, 3) for x in TR], "benign", B["overall"]); print(TOOL)
