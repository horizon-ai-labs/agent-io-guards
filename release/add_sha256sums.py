"""Write SHA256SUMS (sha256sum format) for the inference artifacts of every Horizon-Labs model repo: weights, ONNX files, tokenizer and
config files, and any other top-level/onnx file needed to run the model. Excluded: README.md, .gitattributes, LICENSE, NOTICE, SHA256SUMS
and the code/, eval/, training/ folders (documentation that changes over time). LFS files use the Hub's sha256; small files are hashed
after download.  python release/add_sha256sums.py [--only repo1,repo2]
Regenerate whenever a model file changes (publish scripts: run this after uploading new weights)."""
import hashlib, sys
from huggingface_hub import HfApi, hf_hub_download, CommitOperationAdd
api = HfApi(); only = set(sys.argv[2].split(",")) if len(sys.argv) > 2 and sys.argv[1] == "--only" else None
SKIP = {"README.md", ".gitattributes", "LICENSE", "NOTICE", "SHA256SUMS"}
for m in api.list_models(author="Horizon-Labs"):
    if only and m.id not in only:
        continue
    i = api.model_info(m.id, files_metadata=True); lines = []
    for s in sorted(i.siblings, key=lambda s: s.rfilename):
        f = s.rfilename
        if f in SKIP or f.split("/")[0] in ("code", "eval", "training"):
            continue
        h = s.lfs.sha256 if s.lfs else hashlib.sha256(open(hf_hub_download(m.id, f, revision=i.sha), "rb").read()).hexdigest()
        lines.append(f"{h}  {f}")
    sha = api.create_commit(m.id, operations=[CommitOperationAdd("SHA256SUMS", ("\n".join(lines) + "\n").encode())],
                            commit_message="Add SHA256SUMS for the model files", parent_commit=i.sha).oid
    print(m.id, len(lines), "files", sha[:8], flush=True)
