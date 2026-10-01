"""python release/rerank/publish_rr.py SIZE RELEASE_DIR DATA_STATS.json "msg"   -> Horizon-Labs/multilingual-reranker-{small,base}"""
import json, os, sys
from huggingface_hub import HfApi, CommitOperationAdd
size, d, stats, msg = sys.argv[1:5]
repo = f"Horizon-Labs/multilingual-reranker-{size}"
api = HfApi(); api.create_repo(repo, exist_ok=True)
ops = [CommitOperationAdd(f, f"{d}/{f}") for f in ["config.json", "model.safetensors", "tokenizer.json", "tokenizer_config.json", "special_tokens_map.json"]]
ops += [CommitOperationAdd(f"onnx/{f}", f"{d}/onnx/{f}") for f in ["model.onnx", "model_quantized.onnx"]]
ops.append(CommitOperationAdd("onnx/quantization_check.json", f"{d}/onnx_sweep.json"))
ops.append(CommitOperationAdd("training/val_metrics.json", f"{d}/val_metrics.json"))
t = json.load(open(f"{d}/train_log.json")); t["args"]["data"] = "see code/"; t["args"]["passages"] = "see code/"; t["args"]["out"] = "model"
ops.append(CommitOperationAdd("training/train_log.json", json.dumps(t, indent=1).encode()))
ops.append(CommitOperationAdd("training/data_stats.json", stats))
ops.append(CommitOperationAdd("README.md", f"release/rerank/README_{size}.md"))
ops.append(CommitOperationAdd("eval/this_model.json", f"release/evals/rr_{size}.json"))
ops.append(CommitOperationAdd("eval/baselines.json", "release/evals/rr_baselines.json"))
for p in ["rerank/build_evals.py", "rerank/build_passages.py", "lid/fw_files.json", "rerank/gen_queries.py", "rerank/mine.py", "rerank/subsample.py",
          "rerank/teacher_score_vllm.py", "rerank/train_rr.py", "rerank/eval_rerank.py", "rerank/quant_check.py", "train/export_onnx.py"]:
    ops.append(CommitOperationAdd(f"code/{p}", p))
pats = [x for x in [os.environ.get("HF_TOKEN", ""), os.environ.get("GH_TOKEN", "")] if x]
for o in ops:
    b = o.path_or_fileobj if isinstance(o.path_or_fileobj, bytes) else (open(o.path_or_fileobj, "rb").read() if os.path.getsize(o.path_or_fileobj) < 5e6 else b"")
    assert not any(x.encode() in b for x in pats), o.path_in_repo
print(repo, api.create_commit(repo, operations=ops, commit_message=msg).oid)
