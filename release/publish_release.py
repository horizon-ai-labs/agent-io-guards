"""Upload an assembled release dir (release/run_release_build.sh output) + card + evals + code.
python release/publish_release.py SIZE RELEASE_MODEL_DIR OURS_EVAL_JSON BASELINES_JSON "commit message" """
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
size, d, ev, bl, msg = sys.argv[1:6]
repo = f"Horizon-Labs/prompt-injection-guard-{size}"
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]]
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in ["model.onnx", "model_quantized.onnx"]]
ops.append(CommitOperationAdd("onnx/quantization_check.json", f"{d}/onnx_sweep.json"))
ops.append(CommitOperationAdd("training/val_metrics.json", f"{d}/val_metrics.json"))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("README.md", f"release/{size}/README.md"))
ops.append(CommitOperationAdd("eval/eval_results.json", ev)); ops.append(CommitOperationAdd("eval/baselines_eval_results.json", bl))
for p in ["data/build_v0.py", "data/build_v1.py", "data/build_v2.py", "data/langs.py", "gen/gen_v1.py", "gen/gen_v2.py", "train/train.py",
          "train/evaluate.py", "train/normalizer.py", "train/export_onnx.py", "train/quant_sweep.py", "release/build_release_dir.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
for o in ops:
    if isinstance(o.path_or_fileobj, bytes):
        continue
    s = open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b""
    assert not any(x.encode() in s for x in os.environ.get("LEAK_PATTERNS", "").split(",") if x), o.path_in_repo
info = HfApi().create_commit(repo, operations=ops, commit_message=msg)
print(repo, info.oid)
