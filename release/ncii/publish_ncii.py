"""python release/ncii/publish_ncii.py RELEASE_DIR EVAL_DIR "msg"  -> Horizon-Labs/multilingual-ncii-guard-base
(card: release/ncii/README.md from make_ncii_card.py; afterwards run add_license_notice.py and add_sha256sums.py --only)"""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
d, ev, msg = sys.argv[1:4]
repo = "Horizon-Labs/multilingual-ncii-guard-base"
api = HfApi(); api.create_repo(repo, exist_ok=True)
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]]
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in ["model.onnx", "model_quantized.onnx"]]
ops.append(CommitOperationAdd("onnx/onnx_sweep.json", f"{d}/onnx_sweep.json"))
ops.append(CommitOperationAdd("training/val_metrics.json", f"{d}/val_metrics.json"))
ops.append(CommitOperationAdd("README.md", "release/ncii/README.md"))
ops.append(CommitOperationAdd("eval/results.json", f"{ev}/eval.json"))
for p in ["ncii/gen_ncii.py", "ncii/extract_benign.py", "ncii/build_pool.py", "ncii/judge_ncii.py", "ncii/build_data.py", "ncii/augment.py",
          "ncii/eval_ncii.py", "ncii/build_release_dir.py", "train/train.py", "train/normalizer.py", "train/export_onnx.py", "train/quant_sweep.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
pats = [x for x in [os.environ.get("HF_TOKEN", ""), os.environ.get("GH_TOKEN", "")] if x] + ["awu" + "hrmann"]  # cluster user name
for o in ops:
    b = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert not any(x.encode() in b for x in pats), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
