"""Link the truecase models from the punctuation-restoration cards and Space (README / index.html only)."""
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
api = HfApi()
old = "- It restores punctuation only. It does not change casing or the words.\n"
for size in ("small", "base"):
    repo = f"Horizon-Labs/punctuation-restoration-{size}"; info = api.model_info(repo)
    t = open(hf_hub_download(repo, "README.md", revision=info.sha)).read()
    new = (f"- It restores punctuation only. It does not change casing or the words. For lowercase text (such as ASR output) where you\n"
           f"  also want capital letters, use [multilingual-punctuation-truecase-{size}](https://huggingface.co/Horizon-Labs/multilingual-punctuation-truecase-{size}).\n")
    if "multilingual-punctuation-truecase" in t: print(repo, "already"); continue
    assert t.count(old) == 1, repo
    print(repo, api.create_commit(repo, [CommitOperationAdd("README.md", t.replace(old, new).encode())], parent_commit=info.sha,
                                  commit_message="Card: link the punctuation + truecasing model").oid)
sp = "Horizon-Labs/punctuation-restoration"; info = api.space_info(sp)
h = open(hf_hub_download(sp, "index.html", repo_type="space", revision=info.sha)).read()
o2 = "Restores punctuation only (no casing)."
n2 = 'Restores punctuation only (no casing; for capitals too, try <a href="https://huggingface.co/spaces/Horizon-Labs/multilingual-punctuation-truecase">Punctuation + Truecasing</a>).'
if o2 in h and "multilingual-punctuation-truecase" not in h:
    print(sp, api.create_commit(sp, [CommitOperationAdd("index.html", h.replace(o2, n2).encode())], repo_type="space", parent_commit=info.sha,
                                commit_message="Link the punctuation + truecasing demo").oid)
