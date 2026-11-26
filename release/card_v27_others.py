"""Base and large injection cards after small v2.7: small column -> v2.7 (bold recomputed), macro line, third-party small row, and the
tool-output section regenerated with small v2.7 as the comparison. Reuses release/card_v26_others.py and release/card_v27_small.py.
python release/card_v27_others.py [--dry]"""
import sys
sys.path.insert(0, "release")
import card_v27_small as N           # sets C.E / C.MACRO to the v2.7 eval
import card_v26_others as O
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

C = O.C; C.M26 = N.M27; C.TR_ROW = N.TR_ROW   # section() and the third-party row read these


def section(size):
    s = O.section(size).replace("small v2.6", "small v2.7")
    return s.replace("| **this model** (base v2.2) | small v2.7 | small v2.3 |", "| **this model** (base v2.2) | small v2.7 | small v2.3 |")


if __name__ == "__main__":
    api = HfApi()
    for size in ("base", "large"):
        repo = f"Horizon-Labs/prompt-injection-guard-{size}"; info = api.model_info(repo)
        c = open(hf_hub_download(repo, "README.md", revision=info.sha)).read(); assert "small v2.7" not in c
        c = O.patch_macro(O.patch_tables(c, O.SMALL_COL[size]))
        old = [l for l in c.split("\n") if l.startswith("| **Horizon-Labs/prompt-injection-guard-small (v2.6)**")]; assert len(old) == 1
        c = c.replace(old[0], N.TR_ROW)
        i = c.index("### Tool outputs and terse status lines"); j = c.index("### Third-party multilingual benchmarks")
        c = c[:i] + section(size) + c[j:]
        c = c.replace("## Changelog\n\n", "## Changelog\n\n- **2026-11-26**: card only. Small moved to v2.7 (small column, rows and tool-output comparison updated).\n")
        open(f".tmp/inj_{size}_v27.md", "w").write(c)
        if "--dry" in sys.argv:
            print(size, "ok"); continue
        print(repo, api.create_commit(repo, [CommitOperationAdd("README.md", c.encode())], parent_commit=info.sha,
                                      commit_message="Card: small moved to v2.7").oid)
