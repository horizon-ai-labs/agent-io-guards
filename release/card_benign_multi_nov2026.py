"""Add 'False alarms on real multilingual user prompts' (inj/benign_multi_fpr.py, job 00367b96) to the injection guard small/base cards,
plus eval/benign_multilingual_fpr.json and code/inj/benign_multi_fpr.py. README/eval/code only (not covered by SHA256SUMS)."""
import json
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
api = HfApi(); r = json.load(open("release/evals/inj_benign_multi_fpr.json"))
K = {"small": "9baac730", "base": "8f435d9b"}; V = {"small": "v2.3", "base": "v2.2"}
pc = lambda x: f"{100 * x:.1f}%" if x >= .001 or x == 0 else f"{100 * x:.2f}%"
anchor = "## Built-in obfuscation normalizer"
for size in ("small", "base"):
    other = "base" if size == "small" else "small"; m, o = r[K[size]], r[K[other]]
    row = lambda name, v: (f"| {name} | {pc(v['overall']['fpr50'])} / {pc(v['overall']['fpr90'])} | {pc(v['by_src']['aya']['fpr50'])} / {pc(v['by_src']['aya']['fpr90'])} | "
                           f"{pc(v['by_src']['oasst2']['fpr50'])} / {pc(v['by_src']['oasst2']['fpr90'])} | {pc(v['per_lang']['en']['fpr50'])} / {pc(v['per_lang']['en']['fpr90'])} | "
                           f"{pc(v['per_lang']['pt']['fpr50'])} / {pc(v['per_lang']['pt']['fpr90'])} |")
    sec = f"""### False alarms on real multilingual user prompts (checked November 2026)

How often does the guard flag ordinary user messages? We scored 24,788 real, human-written prompts that were never used in
training: 20,875 from the [Aya dataset](https://huggingface.co/datasets/CohereLabs/aya_dataset) (65 languages) and 3,913 user
turns from [OASST2](https://huggingface.co/datasets/OpenAssistant/oasst2) (14 languages), both Apache-2.0, up to 400 per language
and source. Every prompt is benign, so every flag is a false alarm. Share flagged at threshold 0.5 / 0.9:

| model | all | Aya | OASST2 | English (OASST2) | Portuguese, mostly pt-BR (OASST2) |
|---|---|---|---|---|---|
{row(f"**this model** ({V[size]})", m)}
{row(f"[{other}](https://huggingface.co/Horizon-Labs/prompt-injection-guard-{other}) ({V[other]})", o)}

- No language is above 1.3% at 0.9. The highest rates at 0.5 (about 4%) are in English OASST2 chat turns, and for base also in\n  Chinese ones; non-English languages are not worse than English.
- The flagged prompts are mostly questions about the assistant itself ("do you have any censorship?", "what are your weak
  points?"), URLs, shell commands and rewrite requests. Chat messages that talk *to* the AI about its rules are the hardest
  case, because injections do the same.
- Per-language numbers: `eval/benign_multilingual_fpr.json`; script: `code/inj/benign_multi_fpr.py`.

"""
    repo = f"Horizon-Labs/prompt-injection-guard-{size}"; info = api.model_info(repo)
    t = open(hf_hub_download(repo, "README.md", revision=info.sha)).read()
    if "False alarms on real multilingual user prompts" in t: print(repo, "already"); continue
    assert t.count(anchor) == 1
    ops = [CommitOperationAdd("README.md", t.replace(anchor, sec + anchor).encode()),
           CommitOperationAdd("eval/benign_multilingual_fpr.json", "release/evals/inj_benign_multi_fpr.json"),
           CommitOperationAdd("code/inj/benign_multi_fpr.py", "inj/benign_multi_fpr.py")]
    print(repo, api.create_commit(repo, ops, parent_commit=info.sha, commit_message="Card: false alarms on real multilingual user prompts (Aya, OASST2)").oid)
