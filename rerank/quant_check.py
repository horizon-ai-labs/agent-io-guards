"""int8 (gather-only) ONNX check for a reranker.   python rerank/quant_check.py MODEL_DIR EVAL_DIR
Quantizes onnx/model.onnx -> onnx/model_quantized.onnx (embedding Gather only) and compares with fp32 ONNX on (query,
passage) pairs from the first candidates of 4 eval sets: mean / max |sigmoid difference| and nDCG@10 on those sets.
Writes MODEL_DIR/onnx_sweep.json (same keys as train/quant_sweep.py: gather_only.{mean_diff,max_diff,mb}, n)."""
import glob, json, math, os, sys
import numpy as np, pandas as pd, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

d, ev = sys.argv[1:3]
tok = AutoTokenizer.from_pretrained(d)
sets = [p for p in sorted(glob.glob(f"{ev}/*.parquet")) if os.path.basename(p).split(".")[0] in ("miracl_de", "miracl_zh", "wiki_en", "esci_jp", "rubq")]
data = [(pd.read_parquet(p).head(12)) for p in sets]
so = ort.SessionOptions(); so.intra_op_num_threads = 8


def run(path):
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"]); allp = []; nd = []
    for df in data:
        for q, docs, rels in zip(df["query"], df.docs, df.rels):
            docs = list(docs)[:30]; rels = list(rels)[:30]
            e = tok([q] * len(docs), docs, truncation="only_second", max_length=384, padding=True, return_tensors="np")
            lo = s.run(None, {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]})[0][:, 0]
            allp += list(1 / (1 + np.exp(-lo)))
            order = np.argsort(-lo)[:10]; ideal = sorted(rels, reverse=True)[:10]
            idcg = sum((2 ** r - 1) / math.log2(k + 2) for k, r in enumerate(ideal))
            nd.append(sum((2 ** rels[i] - 1) / math.log2(k + 2) for k, i in enumerate(order)) / idcg if idcg else 0)
    return np.array(allp), float(np.mean(nd))


ref, nd_ref = run(f"{d}/onnx/model.onnx")
quantize_dynamic(f"{d}/onnx/model.onnx", f"{d}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
q, nd_q = run(f"{d}/onnx/model_quantized.onnx")
r = dict(gather_only=dict(mean_diff=float(np.abs(q - ref).mean()), max_diff=float(np.abs(q - ref).max()), ndcg_fp32=nd_ref, ndcg_int8=nd_q,
                          mb=os.path.getsize(f"{d}/onnx/model_quantized.onnx") / 1e6), n=int(len(ref)), best="gather_only")
json.dump(r, open(f"{d}/onnx_sweep.json", "w"), indent=1); print(r, flush=True)
