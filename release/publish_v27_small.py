"""Publish prompt-injection-guard-small v2.7: release dir bc6a3583 + patched card (.tmp/inj_small_v27.md from release/card_v27_small.py)
+ as-shipped eval + tool-output eval + code, then tag v2.7.   python release/publish_v27_small.py [--dry]"""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd

REPO, D = "Horizon-Labs/prompt-injection-guard-small", "/outputs/bc6a3583-b2b0-40f5-8d6c-c0090c0bfec2/model"
EV = "release/evals/inj_small_v27_eval.json"   # sanitized copy of the release job's eval_results.json (no cluster paths)
ops = [CommitOperationAdd(f, f"{D}/{f}") for f in ["config.json", "model.safetensors", "tokenizer_config.json", "special_tokens_map.json"]]
# the release build predates the build_release_dir.py ANSI fix: upload the tokenizer patched by release/ansi_normalizer.py (normalizer
# byte-identical to v2.6.1; checks (a)/(b) passed on 54,952 eval texts)
ops.append(CommitOperationAdd("tokenizer.json", ".tmp/ansi/v27_tokenizer.json"))
ops += [CommitOperationAdd(f"onnx/{f}", f"{D}/onnx/{f}") for f in sorted(os.listdir(f"{D}/onnx"))]
ops.append(CommitOperationAdd("onnx/quantization_check.json", f"{D}/onnx_sweep.json"))
ops.append(CommitOperationAdd("training/val_metrics.json", f"{D}/val_metrics.json"))
t = json.load(open(f"{D}/train_log.json")); t["args"]["data"] = "see code/ (inj/build_v27.py)"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("README.md", ".tmp/inj_small_v27.md"))
ops.append(CommitOperationAdd("eval/eval_results.json", EV))
ops.append(CommitOperationAdd("eval/tool_outputs.json", "release/evals/inj_tool_outputs_v27.json"))
for p in ["train/train.py", "train/evaluate.py", "train/normalizer.py", "train/export_onnx.py", "train/quant_sweep.py", "release/build_release_dir.py",
          "inj/teacher_inj.py", "inj/build_v22.py", "inj/gen_hijack3.py", "inj/gen_benign_cmd.py", "inj/build_v23.py", "inj/build_v24_v25.py",
          "inj/gen_terse.py", "inj/build_v26.py", "inj/build_tool_evals.py", "inj/tool_metrics.py", "inj/gen_tool2.py", "inj/build_v27.py", "release/ansi_normalizer.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
for o in ops:   # leak check (user name / cluster paths) on every small file
    s = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert b"awu" + b"hrmann" not in s and b"/fast/" not in s, o.path_in_repo
print(len(ops), "files")
if "--dry" not in sys.argv:
    api = HfApi(); info = api.create_commit(REPO, operations=ops, commit_message="v2.7: id-like log lines, legitimate-instruction JSON and hidden agent instructions in tool output")
    api.create_tag(REPO, tag="v2.7", revision=info.oid); print(REPO, info.oid, "tagged v2.7")
