"""python release/pii/publish_pii.py SIZE RELEASE_DIR OURS_KEY EXT_EVAL_JSON INDIST_JSON "msg" """
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
size, d, key, ext, indist, msg = sys.argv[1:7]
repo = f"Horizon-Labs/pii-redactor-{size}"
api = HfApi(); api.create_repo(repo, exist_ok=True)
E = json.load(open(ext))
clean = {}
for k, v in E.items():
    if k.startswith("/"):
        if k != key: continue
        k = repo
    clean[k] = v
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                                    "special_tokens_map.json", "redact.py"]]
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in os.listdir(f"{d}/onnx")]
ops.append(CommitOperationAdd("onnx/quantization_check.json", f"{d}/onnx_check.json"))
ops.append(CommitOperationAdd("README.md", f"release/pii/{size}/README.md"))
ops.append(CommitOperationAdd("eval/external_benchmarks.json", json.dumps(clean, indent=1).encode()))
ops.append(CommitOperationAdd("eval/in_distribution.json", indist))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
for p in ["pii/build_pii_v0.py", "pii/train_tok.py", "pii/evaluate_pii.py", "pii/export_onnx_tok.py", "pii/gen_pii.py", "pii/build_pii_v1.py", "pii/build_pii_v3.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
for o in ops:
    if isinstance(o.path_or_fileobj, str) and os.path.getsize(o.path_or_fileobj) < 5e6:
        b = open(o.path_or_fileobj, "rb").read(); assert not any(x.encode() in b for x in os.environ.get("LEAK_PATTERNS", "").split(",") if x), o.path_in_repo
    elif isinstance(o.path_or_fileobj, bytes):
        assert not any(x.encode() in o.path_or_fileobj for x in os.environ.get("LEAK_PATTERNS", "").split(",") if x), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
