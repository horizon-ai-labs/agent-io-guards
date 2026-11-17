"""Add a verified 'Use with LLM Guard' section to the multilingual-toxicity cards (README only). Check: tox/llmguard_check.py, job 154ee6e1."""
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
api = HfApi()
for size in ("base", "small"):
    repo = f"Horizon-Labs/multilingual-toxicity-{size}"
    checked = ("Checked in November 2026 with llm-guard 0.3.16 (input and output scanners, ONNX and PyTorch paths)" if size == "base" else
               "Checked in November 2026 with llm-guard 0.3.16 on the base model, which has the same labels and architecture")
    sec = f"""### Use with LLM Guard

The labels are the Detoxify set that [LLM Guard](https://github.com/protectai/llm-guard)'s `Toxicity` scanners expect, so this model
can replace their English default (`unitary/unbiased-toxic-roberta`) to scan prompts and responses in many languages:

```python
from llm_guard.input_scanners import Toxicity            # llm_guard.output_scanners.Toxicity works the same way
from llm_guard.model import Model

model = Model(path="{repo}", onnx_path="{repo}", onnx_subfolder="onnx",
              pipeline_kwargs={{"top_k": None, "function_to_apply": "sigmoid", "truncation": True, "max_length": 512,
                               "return_token_type_ids": False}})
scanner = Toxicity(model=model, threshold=0.5)           # use_onnx=True for the ONNX files
sanitized, is_valid, risk = scanner.scan("Du bist ein dummer Vollidiot.")   # is_valid == False
```

{checked}. Toxic and harmless test sentences in English, German, Spanish and Russian were all classified correctly.

"""
    i = api.model_info(repo); t = open(hf_hub_download(repo, "README.md", revision=i.sha)).read()
    if "### Use with LLM Guard" in t:
        print(repo, "already"); continue
    assert t.count("## Evaluation") == 1, repo
    print(repo, api.create_commit(repo, [CommitOperationAdd("README.md", t.replace("## Evaluation", sec + "## Evaluation").encode())],
                                  parent_commit=i.sha, commit_message="Card: verified LLM Guard Toxicity scanner usage").oid)
