"""Base and large injection cards after small v2.6: small column -> v2.6 numbers (bold recomputed), macro line, third-party small row,
and a short 'Tool outputs and terse status lines' section (base v2.2 and large v2.1 share the terse false-alarm problem).
Reuses release/card_v26_small.py.   python release/card_v26_others.py [--dry]"""
import re, sys
sys.path.insert(0, "release")
import card_v26_small as C
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

SMALL_COL = {"base": 2, "large": 3}   # cell index of "small (141M)" in the comparison tables
SMALL_LINK = "[small v2.6](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small)"


def patch_tables(t, col):
    out = []
    for line in t.split("\n"):
        if line.startswith("| ") and "|---" not in line and not line.startswith("| Eval set") and not line.startswith("| ROC AUC"):
            cells = [c.strip() for c in line.strip("|").split("|")]; label = cells[0]; new = None
            if label in C.AUC and len(cells) >= 10:
                new = C.f3(C.E[C.AUC[label]]["auc"])
            else:
                for pre, (k, m) in C.ROWS.items():
                    if label.startswith(pre) and len(cells) >= 10:
                        new = f"{C.f3(C.E[k]['f1'])} ({C.E[k]['fpr']:.2f})" if m == "f1" else C.f3(C.E[k]["acc"])
            if new is not None:
                vals = cells[1:]; vals[col - 1] = new; line = "| " + " | ".join([label] + C.rebold(vals)) + " |"
        out.append(line)
    return "\n".join(out)


def patch_macro(t):
    i = t.index("Macro average over the external (non-synthetic) sets above: "); j = t.index("\n", i)
    items = [x.strip().rstrip(".") for x in t[i:j].split(": ", 1)[1].split(" · ")]
    d = {x.rsplit(": ", 1)[0]: float(x.rsplit(": ", 1)[1]) for x in items}; d["small (141M)"] = C.MACRO
    return t[:i] + "Macro average over the external (non-synthetic) sets above: " + " · ".join(
        f"{k}: {v:.3f}" for k, v in sorted(d.items(), key=lambda kv: -kv[1])) + "." + t[j:]


def section(size):
    me = C.MB if size == "base" else C.ML; ver = "v2.2" if size == "base" else "v2.1"
    return f"""### Tool outputs and terse status lines (checked November 2026)

Agents scan tool results after every call, and many are terse machine text. A user
[reported](https://huggingface.co/Horizon-Labs/prompt-injection-guard-small/discussions/1) that our models flag status lines such as
`probe q3481 ok n=11`, also inside MCP results. This model ({size} {ver}) has the same problem; {SMALL_LINK} was retrained to fix it.
Threshold 0.5:

| | **this model** ({size} {ver}) | small v2.6 | small v2.3 |
|---|---|---|---|
| held-out terse status / ack / log lines (575), flagged | {C.pc(me['terse_fpr'])} | {C.pc(C.M26['terse_fpr'])} | {C.pc(C.M23['terse_fpr'])} |
| short attacks inside those envelopes (300), caught | {C.pc(me['terse_pos_rec'])} | {C.pc(C.M26['terse_pos_rec'])} | {C.pc(C.M23['terse_pos_rec'])} |
| [tool-output-injection-bench](https://huggingface.co/datasets/SamJoshua/tool-output-injection-bench) (408), F1 | {C.f3(me['toib_f1'])} | {C.f3(C.M26['toib_f1'])} | {C.f3(C.M23['toib_f1'])} |
| same bench, benign outputs flagged (of 156) | {me['toib_benign_fp']} | {C.M26['toib_benign_fp']} | {C.M23['toib_benign_fp']} |

- If you scan terse tool output (status lines, logs, acknowledgements), use small v2.6 for it, or skip the classifier for status formats
  your own tools emit. Details and caveats: the small model's card.

"""


if __name__ == "__main__":
    api = HfApi()
    for size in ("base", "large"):
        repo = f"Horizon-Labs/prompt-injection-guard-{size}"; info = api.model_info(repo)
        c = open(hf_hub_download(repo, "README.md", revision=info.sha)).read(); assert "Tool outputs and terse status lines" not in c
        c = patch_macro(patch_tables(c, SMALL_COL[size]))
        old = [l for l in c.split("\n") if l.startswith("| **Horizon-Labs/prompt-injection-guard-small (v2.3)**")]; assert len(old) == 1
        c = c.replace(old[0], C.TR_ROW)
        c = c.replace("### Third-party multilingual benchmarks (checked October 2026)", section(size) + "### Third-party multilingual benchmarks (checked October 2026)")
        c = c.replace("## Changelog\n\n", "## Changelog\n\n- **2026-11-25**: card only. Small moved to v2.6 (small column and rows updated); added the tool-output section.\n")
        open(f".tmp/inj_{size}_v26.md", "w").write(c)
        if "--dry" in sys.argv:
            print(size, "ok"); continue
        print(repo, api.create_commit(repo, [CommitOperationAdd("README.md", c.encode())], parent_commit=info.sha,
                                      commit_message="Card: small moved to v2.6; tool-output / terse status line results").oid)
