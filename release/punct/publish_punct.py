"""python release/punct/publish_punct.py SIZE RELEASE_DIR "msg"   -> Horizon-Labs/punctuation-restoration-{small,base}"""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
size, d, msg = sys.argv[1:4]
repo = f"Horizon-Labs/punctuation-restoration-{size}"
api = HfApi(); api.create_repo(repo, exist_ok=True)
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                                   "special_tokens_map.json"]]
ops.append(CommitOperationAdd("punctuate.py", "release/punct/punctuate.py"))
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in ["model.onnx", "model_quantized.onnx"] if os.path.exists(f"{d}/onnx/{f}")]
ops.append(CommitOperationAdd("onnx/onnx_check.json", f"{d}/onnx_check.json"))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("training/data_stats.json", "release/evals/punct_data_stats.json"))
ops.append(CommitOperationAdd("README.md", f"release/punct/README_{size}.md"))
ops.append(CommitOperationAdd("eval/this_model.json", f"release/evals/punct_{size}.json"))
ops.append(CommitOperationAdd("eval/baselines.json", "release/evals/punct_baselines.json"))
for p in ["punct/common.py", "punct/build_data.py", "punct/build_evals.py", "punct/train_punct.py", "punct/eval_punct.py", "punct/eval_pcs.py",
          "emb/build_texts.py", "lid/fw_files.json", "pii/export_onnx_tok.py", "punct/gen_spoken.py", "punct/build_spoken.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
pats = [x for x in [os.environ.get("HF_TOKEN", ""), os.environ.get("GH_TOKEN", "")] if x]
for o in ops:
    b = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert not any(x.encode() in b for x in pats), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
