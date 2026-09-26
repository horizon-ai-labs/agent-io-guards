"""python release/zeroshot/publish_zs.py SIZE RELEASE_DIR "msg" """
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
size, d, msg = sys.argv[1:4]
repo = f"Horizon-Labs/multilingual-zeroshot-{size}"
api = HfApi(); api.create_repo(repo, exist_ok=True)
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]]
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in sorted(os.listdir(f"{d}/onnx"))]
ops.append(CommitOperationAdd("onnx/quantization_check.json", f"{d}/onnx_sweep.json"))
ops.append(CommitOperationAdd("training/val_metrics.json", f"{d}/val_metrics.json"))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("README.md", f"release/zeroshot/{size}/README.md"))
ops.append(CommitOperationAdd("eval/this_model.json", f"release/evals/zs_{size}.json"))
ops.append(CommitOperationAdd("eval/baselines.json", "release/evals/zs_baselines.json"))
ops.append(CommitOperationAdd("eval/native_labels.json", "release/evals/zs_native_labels.json"))
for p in ["zeroshot/gen_zeroshot.py", "zeroshot/gen_zeroshot_v2.py", "zeroshot/translate_nli.py", "zeroshot/translate_labelsets.py", "zeroshot/translate_labels.py", "zeroshot/build_zs_data.py", "zeroshot/build_evals.py",
          "zeroshot/evaluate_zs.py", "train/train.py", "train/export_onnx.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
pats = [x for x in os.environ.get("LEAK_PATTERNS", "").split(",") + [os.environ.get("HF_TOKEN", ""), os.environ.get("GH_TOKEN", "")] if x]
for o in ops:
    b = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert not any(x.encode() in b for x in pats), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
