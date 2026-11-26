"""Publish the ANSI-escape normalizer rule (release/ansi_normalizer.py) to the three injection repos: tokenizer.json + card notes + code,
tags small v2.6.1 / base v2.2.1 / large v2.1.1 (weights unchanged). Numbers: SamJoshua/terse-tool-output-heldout (rev 16d01ee2, test-only),
scored on CPU with each release's int8 ONNX file, old vs new tokenizer (.tmp/ansi/score.py, 2026-11-26).   python release/publish_ansi_v261.py [--dry]"""
import sys
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

ITEM = ("3. It **strips invisible and formatting characters**: zero-width characters, bidi controls, soft hyphens, and combining\n"
        "   underline and overlay marks.\n")
ANSI_ITEM = ITEM + ("4. It **removes ANSI terminal escape sequences** (colours, cursor movement, hyperlinks) before anything else. Shell tools emit them,\n"
                    "   and coloured status lines were flagged far more often than the same lines in plain text. Text without an ESC character is\n"
                    "   unchanged.\n")
HELD = {   # benign flagged (312) / ANSI-wrapped benign flagged (23) / attacks caught (88): old tokenizer -> new tokenizer
    "small": ("48", "29", "20", "1", "85", "84"), "base": ("90", "84", "16", "10", "87", "87"), "large": ("99", "99", "9", "9", "87", "87")}
VER = {"small": ("v2.6.1", "v2.6"), "base": ("v2.2.1", "v2.2"), "large": ("v2.1.1", "v2.1")}


def log(size):
    h = HELD[size]; v, old = VER[size]
    return (f"- **{v}** (2026-11-26): tokenizer only, weights unchanged. The built-in normalizer now removes ANSI terminal escape sequences.\n"
            f"  On [SamJoshua/terse-tool-output-heldout](https://huggingface.co/datasets/SamJoshua/terse-tool-output-heldout) (400 terse tool outputs,\n"
            f"  test-only): benign lines flagged {h[0]} -> {h[1]} of 312, of which ANSI-coloured {h[2]} -> {h[3]} of 23; attacks caught {h[4]} -> {h[5]} of 88.\n"
            f"  All our other eval sets contain no ANSI escapes, so their scores are unchanged.\n")


SMALL_NOTE = ("- **Since v2.6.1**, ANSI colour codes are stripped by the tokenizer (see the normalizer section). On SamJoshua's second held-out\n"
              "  set ([terse-tool-output-heldout](https://huggingface.co/datasets/SamJoshua/terse-tool-output-heldout), 400 rows, 12 tool formats,\n"
              "  11 wrappers we never trained on) this version flags 29 of 312 benign lines (v2.3: 101) and catches 84 of 88 attacks (v2.3: 86).\n"
              "  One attack that was caught only because of its colour codes now scores 0.48, the same as its plain text. Most remaining false\n"
              "  alarms are id-like lines: bare image ids, `sha256:` digests, `#7 CACHED`, `PASS`, timings.\n")

if __name__ == "__main__":
    api = HfApi()
    for size in ("small", "base", "large"):
        repo = f"Horizon-Labs/prompt-injection-guard-{size}"; info = api.model_info(repo)
        c = open(hf_hub_download(repo, "README.md", revision=info.sha)).read()
        assert c.count(ITEM) == 1 and "removes ANSI terminal escape" not in c, size
        c = c.replace(ITEM, ANSI_ITEM).replace("## Changelog\n\n", "## Changelog\n\n" + log(size))
        if size == "small":
            a = "- Instructions hidden in JSON fields are still the weakest carrier on that bench."; assert c.count(a) == 1
            c = c.replace(a, SMALL_NOTE + a)
        ops = [CommitOperationAdd("README.md", c.encode()), CommitOperationAdd("tokenizer.json", f".tmp/ansi/{size}_tokenizer.json"),
               CommitOperationAdd("code/train/normalizer.py", "train/normalizer.py"), CommitOperationAdd("code/release/ansi_normalizer.py", "release/ansi_normalizer.py")]
        if "--dry" in sys.argv:
            open(f".tmp/ansi/{size}_README.md", "w").write(c); print(size, "dry ok"); continue
        oid = api.create_commit(repo, ops, parent_commit=info.sha, commit_message=f"{VER[size][0]}: strip ANSI escape sequences in the tokenizer normalizer").oid
        api.create_tag(repo, tag=VER[size][0], revision=oid); print(repo, oid, VER[size][0])
