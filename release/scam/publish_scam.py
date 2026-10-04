"""python release/scam/publish_scam.py RELEASE_DIR "msg"   -> Horizon-Labs/phishing-scam-detector-base"""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
d, msg = sys.argv[1:3]
repo = "Horizon-Labs/phishing-scam-detector-base"
api = HfApi(); api.create_repo(repo, exist_ok=True)
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]]
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in ["model.onnx", "model_quantized.onnx"] if os.path.exists(f"{d}/onnx/{f}")]
ops.append(CommitOperationAdd("onnx/quantization_check.json", f"{d}/onnx_sweep.json"))
ops.append(CommitOperationAdd("training/val_metrics.json", f"{d}/val_metrics.json"))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("training/data_stats.json", "release/evals/scam_data_stats.json"))
ops.append(CommitOperationAdd("README.md", "release/scam/README_base.md"))
for a, b in [("eval/this_model.json", "scam_base.json"), ("eval/seeds.json", "scam_seeds.json"), ("eval/baselines.json", "scam_baselines.json")]:
    ops.append(CommitOperationAdd(a, f"release/evals/{b}"))
for p in ["scam/common.py", "scam/build_evals.py", "scam/eval_scam.py", "scam/teacher_scam.py", "scam/gen_scam.py", "scam/build_scam.py",
          "sentiment/train_sent.py", "sentiment/gen_sent.py", "emb/build_texts.py", "train/export_onnx.py", "train/quant_sweep.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
pats = [x for x in [os.environ.get("HF_TOKEN", ""), os.environ.get("GH_TOKEN", "")] if x]
for o in ops:
    b = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert not any(x.encode() in b for x in pats), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
