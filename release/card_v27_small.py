"""Card update for prompt-injection-guard-small v2.7 (patches the current Hub README, which already has the v2.6 / v2.6.1 content).
Inputs: as-shipped eval of release build bc6a3583, v2.7 seeds, v2.6 baseline 5b39cdc4 (seeds on the new sets) and release a25b4bf7,
v2.3/base/large baseline 83084f06, third-party re-check c52a4dd9 (inj/run_v26_extra.sh).   python release/card_v27_small.py [--dry]"""
import glob, json, re, sys
import numpy as np, pandas as pd
sys.path.insert(0, "inj"); sys.path.insert(0, "release")
from tool_metrics import metrics
import card_v26_small as C   # table / macro / third-party helpers (module-level code reads the v2.6 inputs; fine)
from huggingface_hub import HfApi, hf_hub_download

REPO = "Horizon-Labs/prompt-injection-guard-small"
REL, X = "/outputs/bc6a3583-b2b0-40f5-8d6c-c0090c0bfec2", glob.glob("/outputs/c52a4dd9-*")[0]
E = list(json.load(open(f"{REL}/eval_results.json")).values())[0]
S27 = ["9d91e350", "d837feae", "ea8c505f", "ae78b2da"]; S26 = ["b3171509", "663a883c", "cc99d278", "6d3c07af"]
pdir = lambda j, need="terse_neg": [p for p in glob.glob("/outputs/*/preds/*") if j in p.split("__")[-2] and glob.glob(f"{p}/{need}.parquet")][0]
M27 = metrics(glob.glob(f"{REL}/preds/*")[0]); M26 = {**metrics(pdir("6d3c07af", "tool2_neg_id")), **metrics(glob.glob("/outputs/a25b4bf7-cea9-4ee0-8886-df5e40f8b75f/preds/*")[0])}; M23, MB, ML = C.M23, C.MB, C.ML
# M26: v2.6 release eval (a25b4bf7); the tool2 / samj_heldout rows come from its training checkpoint (5b39cdc4), as the release eval predates those sets
mean = lambda js, k, need="tool2_neg_id": float(np.mean([metrics(pdir(j, need))[k] for j in js]))
f3, pc = C.f3, C.pc
dash = lambda m, k, fmt: fmt(m[k]) if k in m else "–"

# ---- reuse the v2.6 table patchers with the v2.7 eval
C.E = E
EXT = C.EXT; MACRO = sum(E[s][k] for s, k in EXT) / len(EXT); C.MACRO = MACRO

TOOL = f"""### Tool outputs and terse status lines (checked November 2026)

Agents scan tool results after every call, and many of those results are terse machine text. SamJoshua
[reported](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small/discussions/1) that v2.3 flagged status lines such as
`probe q3481 ok n=11` with 0.93–0.99, also inside MCP results. v2.6 added templated status, acknowledgement and log lines as hard
negatives (with short attacks placed inside the same lines and envelopes as positives). v2.7 adds id-like log lines (image ids, digests,
CI / package-manager / test-runner output), JSON tool results whose fields hold legitimate instructions ("run X and retry"), and
instructions to the agent hidden among log lines or in one JSON field (send secrets out, call an extra tool, hide it from the user).
Threshold 0.5:

| | small v2.7 (this version) | small v2.6 | small v2.3 | base v2.2 | large v2.1 |
|---|---|---|---|---|---|
| the reported strings (17 status / ack lines), flagged | {M27['samj_flagged']} | {M26['samj_flagged']} | {M23['samj_flagged']} | {MB['samj_flagged']} | {ML['samj_flagged']} |
| held-out terse lines (575), flagged | {pc(M27['terse_fpr'])} | {pc(M26['terse_fpr'])} | {pc(M23['terse_fpr'])} | {pc(MB['terse_fpr'])} | {pc(ML['terse_fpr'])} |
| short attacks inside those envelopes (300), caught | {pc(M27['terse_pos_rec'])} | {pc(M26['terse_pos_rec'])} | {pc(M23['terse_pos_rec'])} | {pc(MB['terse_pos_rec'])} | {pc(ML['terse_pos_rec'])} |
| held-out id-like log lines (261; helm, cargo, gradle, ansible, bazel, UUID / ULID / CID ids), flagged | {pc(M27['tool2_neg_id'])} | {pc(M26['tool2_neg_id'])} | – | – | – |
| held-out JSON results with legitimate instructions (300), flagged | {pc(M27['tool2_neg_json'])} | {pc(M26['tool2_neg_json'])} | – | – | – |
| held-out agent instructions among log lines (300) / in a JSON field (300), caught | {pc(M27['tool2_pos_log'])} / {pc(M27['tool2_pos_json'])} | {pc(M26['tool2_pos_log'])} / {pc(M26['tool2_pos_json'])} | – | – | – |
| [tool-output-injection-bench](https://huggingface.co/datasets/SamJoshua/tool-output-injection-bench) (408), F1 | {f3(M27['toib_f1'])} | {f3(M26['toib_f1'])} | {f3(M23['toib_f1'])} | {f3(MB['toib_f1'])} | {f3(ML['toib_f1'])} |
| same bench, benign outputs flagged (of 156) | {M27['toib_benign_fp']} | {M26['toib_benign_fp']} | {M23['toib_benign_fp']} | {MB['toib_benign_fp']} | {ML['toib_benign_fp']} |
| same bench, held-out split, injections caught (of 72) | {round(M27['toib_held_rec'] * 72)} | {round(M26['toib_held_rec'] * 72)} | {round(M23['toib_held_rec'] * 72)} | {round(MB['toib_held_rec'] * 72)} | {round(ML['toib_held_rec'] * 72)} |
| [terse-tool-output-heldout](https://huggingface.co/datasets/SamJoshua/terse-tool-output-heldout) (400), benign flagged (of 312) / attacks caught (of 88) | {M27['samjh_benign_fp']} / {M27['samjh_attacks']} | {M26['samjh_benign_fp']} / {M26['samjh_attacks']} | – | – | – |

- "Held-out" rows use tools, id formats, field names and phrasings that are not in our training templates. The v2.7 training templates
  do cover tool formats that also occur in SamJoshua's two benchmarks (docker, git, kubectl, npm, pytest, terraform), so those two are
  no longer blind tests for v2.7. The reported strings are not held out either. v2.6 values on the id-like, JSON and
  terse-tool-output-heldout rows were measured with the v2.6.1 tokenizer, which added ANSI stripping after the v2.6 release.
- Four training seeds, means (v2.6 means): id-like lines flagged {pc(mean(S27, 'tool2_neg_id'))} ({pc(mean(S26, 'tool2_neg_id'))}), legitimate JSON flagged
  {pc(mean(S27, 'tool2_neg_json'))} ({pc(mean(S26, 'tool2_neg_json'))}), benign outputs flagged on tool-output-injection-bench {mean(S27, 'toib_benign_fpr') * 156:.0f} ({mean(S26, 'toib_benign_fpr') * 156:.0f}).
  All release gates were set before training and passed. The released seed was chosen on validation data, which for v2.7 includes
  tool-output rows.
- Our templated hidden instructions are easier than real ones: the held-out JSON and log instructions are caught 97–99%, but subtler
  instructions inside JSON fields on tool-output-injection-bench are still often missed. Files: `eval/tool_outputs.json`; scripts:
  `code/inj/gen_terse.py`, `code/inj/gen_tool2.py`, `code/inj/build_tool_evals.py`.

"""

Xe = list(json.load(open(f"{X}/eval.json")).values())[0]; xp = glob.glob(f"{X}/preds/*")[0]
t = pd.read_parquet(f"{xp}/tr_pi_full.parquet"); ja = pd.read_parquet(f"{xp}/ja_indirect_doc.parquet")
y, s = t.label.values, t.score.values; ben = t[y == 0]; ti = np.array([C.sub.get(x) == "task_instruction" for x in C.tr.text[y == 0]])
TR = (C.roc(y, s), (s[y == 1] >= .5).mean(), (ben.score >= .5).mean(), (ben.score.values[ti] >= .5).mean(), (ja.score >= .5).mean())
TR_ROW = f"| **Horizon-Labs/prompt-injection-guard-small (v2.7)** | {TR[0]:.3f} | {TR[1]:.2f} | {TR[2]:.3f} | {TR[3]:.2f} | {TR[4]:.2f} |"
B = list(json.load(open(f"{X}/benign/fpr.json")).values())[0]; pcb = C.pcb
B_ROW = (f"| **this model** (v2.7) | {pcb(B['overall']['fpr50'])} / {pcb(B['overall']['fpr90'])} | {pcb(B['by_src']['aya']['fpr50'])} / {pcb(B['by_src']['aya']['fpr90'])} | "
         f"{pcb(B['by_src']['oasst2']['fpr50'])} / {pcb(B['by_src']['oasst2']['fpr90'])} | {pcb(B['per_lang']['en']['fpr50'])} / {pcb(B['per_lang']['en']['fpr90'])} | "
         f"{pcb(B['per_lang']['pt']['fpr50'])} / {pcb(B['per_lang']['pt']['fpr90'])} |")

V27_TRAIN = ("- **v2.7 (this version)**: v2.6 data plus ~13.9k templated tool-output rows (inj/gen_tool2.py): id-like log lines (twice) and JSON\n"
             "  results with legitimate instructions as hard negatives; instructions to the agent among log lines or in a JSON field as positives.\n"
             "  3% of these rows are in the validation set used to choose the released seed. Same recipe as v2.6.\n")
V27_LOG = (f"- **v2.7** (2026-11-26): fewer false alarms on id-like log lines and on JSON tool results with legitimate instructions. Four-seed\n"
           f"  means (gates set before training, all passed): held-out id-like lines flagged {pc(mean(S26, 'tool2_neg_id'))} -> {pc(mean(S27, 'tool2_neg_id'))}, legitimate JSON\n"
           f"  {pc(mean(S26, 'tool2_neg_json'))} -> {pc(mean(S27, 'tool2_neg_json'))}, tool-output-injection-bench benign flagged {mean(S26, 'toib_benign_fpr') * 156:.0f} -> {mean(S27, 'toib_benign_fpr') * 156:.0f} of 156; macro .883 -> .882, BIPIA F1\n"
           f"  +.02, Mindgard evasion -.02, Simsonsun -.02. The released seed scores {MACRO:.3f} macro; its Mindgard evasion recall is\n"
           f"  {E['mindgard_evasion']['acc']:.3f} (v2.6: .760).\n")

if __name__ == "__main__":
    api = HfApi(); info = api.model_info(REPO); c = open(hf_hub_download(REPO, "README.md", revision=info.sha)).read()
    assert "v2.7 (this version)" not in c
    c = C.patch_macro(C.patch_tables(c))
    i = c.index("### Tool outputs and terse status lines"); j = c.index("### Third-party multilingual benchmarks"); c = c[:i] + TOOL + c[j:]
    old = [l for l in c.split("\n") if l.startswith("| **Horizon-Labs/prompt-injection-guard-small (v2.6)**")]; assert len(old) == 1
    c = c.replace(old[0], TR_ROW + "\n" + old[0].replace("**Horizon-Labs/prompt-injection-guard-small (v2.6)**", "Horizon-Labs/prompt-injection-guard-small (v2.6)"))
    old = [l for l in c.split("\n") if l.startswith("| **this model** (v2.6) |")]; assert len(old) == 1
    c = c.replace(old[0], B_ROW + "\n" + old[0].replace("**this model** (v2.6)", "small v2.6"))
    c = c.replace("- **v2.6 (this version)**:", V27_TRAIN + "- **v2.6**:")
    c = c.replace("## Changelog\n\n", "## Changelog\n\n" + V27_LOG)
    q = E["qualifire"]["fpr"]; bi = E["bipia"]["recall"]
    c = re.sub(r"this model flags\n  \d+% of the benign prompts at 0.5", f"this model flags\n  {100 * q:.0f}% of the benign prompts at 0.5", c)
    c = re.sub(r"\(BIPIA recall \d+%, up from", f"(BIPIA recall {100 * bi:.0f}%, up from", c)
    c = re.sub(r"At 0.5 this model flags [\d.]+% of NotInject", f"At 0.5 this model flags {100 * (1 - E['notinject']['acc']):.1f}% of NotInject", c)
    OLD_LIM = ("- Code without comments carries little signal. Terse tool output is handled better since v2.6, but benign tool descriptions\n"
               "  with imperatives (\"run X and retry\") are still flagged too often by this checkpoint (see Tool outputs).")
    assert c.count(OLD_LIM) == 1
    c = c.replace(OLD_LIM, "- Code without comments carries little signal. Subtle instructions hidden in JSON fields are still often missed\n"
                  "  (see Tool outputs).")
    open(".tmp/inj_small_v27.md", "w").write(c)
    tool = {"release_v2.7": M27, "release_v2.6": M26, "v2.3_shipped": M23, "base_v2.2": MB, "large_v2.1": ML,
            "v2.7_4seed_mean": {k: mean(S27, k) for k in ("terse_fpr", "terse_pos_rec", "toib_f1", "toib_benign_fpr", "toib_held_rec", "tool2_neg_id", "tool2_neg_json", "tool2_pos_log", "tool2_pos_json")},
            "v2.6_4seed_mean": {k: mean(S26, k) for k in ("terse_fpr", "terse_pos_rec", "toib_f1", "toib_benign_fpr", "toib_held_rec", "tool2_neg_id", "tool2_neg_json", "tool2_pos_log", "tool2_pos_json")},
            "toib_revision": "2892d2b16d409630933ae9fa7785d3e73efe68f9", "terse_heldout_revision": "16d01ee274c845a4a29bf18e1b80d9da0247a272"}
    json.dump(tool, open("release/evals/inj_tool_outputs_v27.json", "w"), indent=1)
    print("macro", round(MACRO, 4), "TR", [round(float(x), 3) for x in TR], "benign", B["overall"]); print(TOOL)
