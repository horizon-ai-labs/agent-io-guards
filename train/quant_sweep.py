"""Try int8 quantization settings for an exported ONNX classifier; keep the best one.

python train/quant_sweep.py MODEL_DIR EVAL_DIR  -> MODEL_DIR/onnx/model_quantized.onnx (best config), sweep.json
Agreement is measured against the fp32 ONNX model on up to 400 real eval texts.
"""
import glob, json, os, sys, time
import numpy as np, pandas as pd, onnx, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoTokenizer

d, ev = sys.argv[1], sys.argv[2]
tok = AutoTokenizer.from_pretrained(d)
texts = []
for p in sorted(glob.glob(f"{ev}/*.parquet")):
    df = pd.read_parquet(p)
    texts += df.sample(min(len(df), 30), random_state=0).text.str[:3000].tolist()
texts = texts[:400]
so = ort.SessionOptions(); so.intra_op_num_threads = 8


def probs(path):
    s = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
    out = []
    t = time.time()
    for x in texts:
        e = tok(x, truncation=True, max_length=1024, return_tensors="np")
        lo = s.run(None, {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]})[0][0]
        out.append(1 / (1 + np.exp(lo[0] - lo[1])))
    return np.array(out), (time.time() - t) / len(texts)


ref, t_ref = probs(f"{d}/onnx/model.onnx")
m = onnx.load(f"{d}/onnx/model.onnx", load_external_data=False)
names = [n.name for n in m.graph.node]
matmuls = [n.name for n in m.graph.node if n.op_type in ("MatMul", "Gemm")]
head = [n for n in matmuls if any(k in n.lower() for k in ("classifier", "head", "dense"))]
print(len(matmuls), "matmuls; head-like:", head[:10], flush=True)
# Activation-quantizing configs (dynamic MatMul int8) all failed on mmBERT (agreement <= 94.5%, 2026-09-23):
# ModernBERT activations have outliers. Weight-only schemes keep activations in fp32.
configs = {
    "gather_only": dict(op_types_to_quantize=["Gather"]),
}
res = {"fp32_ms": t_ref * 1000}
best = None
for name, kw in configs.items():
    out = f"{d}/onnx/q_{name}.onnx"
    kw = dict(kw); wt = kw.pop("weight_type", QuantType.QInt8)
    try:
        quantize_dynamic(f"{d}/onnx/model.onnx", out, weight_type=wt, **kw)
        p, t = probs(out)
    except Exception as e:
        print(name, "failed", e, flush=True); continue
    r = dict(max_diff=float(np.abs(p - ref).max()), mean_diff=float(np.abs(p - ref).mean()),
             agree=float(((p > 0.5) == (ref > 0.5)).mean()), ms=t * 1000, mb=os.path.getsize(out) / 1e6)
    res[name] = r
    print(name, r, flush=True)
    if best is None or r["mean_diff"] < res[best]["mean_diff"]:
        best = name
# gather int8 + MatMul 4-bit weight-only (block-wise), activations fp32
try:
    from onnxruntime.quantization.matmul_nbits_quantizer import MatMulNBitsQuantizer
    for bits_name, src in [("gather_int8_matmul_q4", f"{d}/onnx/q_gather_only.onnx")]:
        if not os.path.exists(src):
            continue
        mm = onnx.load(src)
        q = MatMulNBitsQuantizer(mm, block_size=32, is_symmetric=True)
        q.process()
        out = f"{d}/onnx/q_{bits_name}.onnx"
        q.model.save_model_to_file(out, use_external_data_format=False)
        p, t = probs(out)
        r = dict(max_diff=float(np.abs(p - ref).max()), mean_diff=float(np.abs(p - ref).mean()),
                 agree=float(((p > 0.5) == (ref > 0.5)).mean()), ms=t * 1000, mb=os.path.getsize(out) / 1e6)
        res[bits_name] = r; print(bits_name, r, flush=True)
        if r["agree"] >= 0.995 and r["mb"] < res[best]["mb"]:
            best = bits_name
except Exception as e:
    print("nbits failed", e, flush=True)
# fp16 (for WebGPU / GPU users), saved separately as model_fp16.onnx
try:
    from onnxconverter_common import float16
    m16 = float16.convert_float_to_float16(onnx.load(f"{d}/onnx/model.onnx"), keep_io_types=True)
    onnx.save(m16, f"{d}/onnx/model_fp16.onnx")
    p, t = probs(f"{d}/onnx/model_fp16.onnx")
    res["fp16"] = dict(max_diff=float(np.abs(p - ref).max()), agree=float(((p > 0.5) == (ref > 0.5)).mean()),
                       ms=t * 1000, mb=os.path.getsize(f"{d}/onnx/model_fp16.onnx") / 1e6)
    print("fp16", res["fp16"], flush=True)
    if res["fp16"]["agree"] < 0.995:
        os.remove(f"{d}/onnx/model_fp16.onnx")
except Exception as e:
    print("fp16 failed", e, flush=True)
json.dump(dict(res, best=best), open(f"{d}/onnx_sweep.json", "w"), indent=1)
print("best", best, res.get(best), flush=True)
if best and res[best]["agree"] >= 0.995:
    os.replace(f"{d}/onnx/q_{best}.onnx", f"{d}/onnx/model_quantized.onnx")
else:
    print("no int8 config agrees >= 99%; removing model_quantized.onnx", flush=True)
    if os.path.exists(f"{d}/onnx/model_quantized.onnx"):
        os.remove(f"{d}/onnx/model_quantized.onnx")
for f in glob.glob(f"{d}/onnx/q_*.onnx"):
    os.remove(f)
