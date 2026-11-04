"""Add the synthetic-data language caveat (eval-suite language-ID check, 2026-11-04) to the prompt-injection-guard cards (README only)."""
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
api = HfApi()
old = "  outside that list are untested.\n"
new = old + """- The language of each synthetic example is the language the generator was asked for. A language-ID check of the
  held-out synthetic sets found 72-93% of non-English rows really in that language (lowest for short direct messages
  in Dutch, Hebrew, Hindi and Bengali, which were often written in English). Per-language synthetic scores are
  therefore approximate; see the `lang_detected` column of the
  [eval suite](https://huggingface.co/datasets/Horizon-Labs/prompt-injection-eval-suite).
"""
for size in ("small", "base", "large"):
    repo = f"Horizon-Labs/prompt-injection-guard-{size}"; info = api.model_info(repo)
    t = open(hf_hub_download(repo, "README.md", revision=info.sha)).read()
    if "language-ID check of the" in t: print(repo, "already"); continue
    assert t.count(old) == 1, repo
    c = api.create_commit(repo, [CommitOperationAdd("README.md", t.replace(old, new).encode())], parent_commit=info.sha,
                          commit_message="Card: note on the language labels of the synthetic data (language-ID check)")
    print(repo, c.oid)
