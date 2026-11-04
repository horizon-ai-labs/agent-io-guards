"""python release/truecase/publish_tc.py SIZE RELEASE_DIR "msg"   -> Horizon-Labs/multilingual-punctuation-truecase-{small,base}
(card: release/truecase/README_{size}.md from make_tc_card.py; afterwards run add_license_notice.py and add_sha256sums.py --only)"""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
size, d, msg = sys.argv[1:4]
repo = f"Horizon-Labs/multilingual-punctuation-truecase-{size}"
api = HfApi(); api.create_repo(repo, exist_ok=True)
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json",
                                                   "special_tokens_map.json"]]
ops.append(CommitOperationAdd("truecase.py", "release/truecase/truecase.py"))
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in ["model.onnx", "model_quantized.onnx"]]
ops.append(CommitOperationAdd("onnx/onnx_check.json", f"{d}/onnx_check.json"))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("training/data_stats.json", "release/evals/punct_data_stats.json"))
ops.append(CommitOperationAdd("README.md", f"release/truecase/README_{size}.md"))
ops.append(CommitOperationAdd("eval/this_model.json", f"release/evals/tc_{size}.json"))
ops.append(CommitOperationAdd("eval/baselines.json", "release/evals/tc_baselines.json"))
if os.path.exists("release/evals/tc_boundary.json"):
    ops.append(CommitOperationAdd("eval/boundary_decoding.json", "release/evals/tc_boundary.json"))
for p in ["punct/common.py", "punct/build_data.py", "punct/build_evals.py", "punct/gen_spoken.py", "punct/build_spoken.py", "emb/build_texts.py",
          "lid/fw_files.json", "truecase/common_tc.py", "truecase/tc_metrics.py", "truecase/train_tc.py", "truecase/eval_tc.py",
          "truecase/pcs_preds.py", "truecase/boundary_check.py", "truecase/build_release_dir.py", "pii/export_onnx_tok.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
pats = [x for x in [os.environ.get("HF_TOKEN", ""), os.environ.get("GH_TOKEN", "")] if x] + ["awu" + "hrmann"]  # cluster user name
for o in ops:
    b = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert not any(x.encode() in b for x in pats), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
