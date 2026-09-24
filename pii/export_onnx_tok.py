"""Export a token-classification model to ONNX (fp32 + int8-embedding) and check agreement with PyTorch.

python pii/export_onnx_tok.py MODEL_DIR EVAL_PARQUET  -> MODEL_DIR/onnx/{model,model_quantized}.onnx, onnx_check.json
Only the embedding Gather is quantized: activation-quantized int8 breaks mmBERT (see train/quant_sweep.py).
"""
import json, os, sys, time
import numpy as np, pandas as pd, torch, onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
from transformers import AutoModelForTokenClassification, AutoTokenizer

d, evp = sys.argv[1], sys.argv[2]
os.makedirs(f"{d}/onnx", exist_ok=True)
tok = AutoTokenizer.from_pretrained(d)
model = AutoModelForTokenClassification.from_pretrained(d, attn_implementation="eager").eval()
enc = tok(["Anna Müller, anna@example.org, +49 30 1234567", "key=sk-live-abc123 on 10.0.0.1"], return_tensors="pt", padding=True)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


with torch.no_grad():
    torch.onnx.export(Wrap(model), (enc["input_ids"], enc["attention_mask"]), f"{d}/onnx/model.onnx",
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                    "logits": {0: "batch", 1: "seq"}}, opset_version=17, dynamo=False)
quantize_dynamic(f"{d}/onnx/model.onnx", f"{d}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])

texts = pd.read_parquet(evp).sample(200, random_state=0).text.str[:2000].tolist()
so = ort.SessionOptions(); so.intra_op_num_threads = 8
res = {}
ref = []
with torch.no_grad():
    for t in texts:
        e = tok(t, truncation=True, max_length=512, return_tensors="pt")
        ref.append(model(**e).logits[0].argmax(-1).numpy())
for f in ["model.onnx", "model_quantized.onnx"]:
    s = ort.InferenceSession(f"{d}/onnx/{f}", so, providers=["CPUExecutionProvider"])
    agree, n, t0 = 0, 0, time.time()
    for t, r in zip(texts, ref):
        e = tok(t, truncation=True, max_length=512, return_tensors="np")
        p = s.run(None, {"input_ids": e["input_ids"], "attention_mask": e["attention_mask"]})[0][0].argmax(-1)
        agree += (p == r).sum(); n += len(r)
    res[f] = dict(token_agreement=float(agree / n), ms_per_doc=(time.time() - t0) / len(texts) * 1000,
                  mb=os.path.getsize(f"{d}/onnx/{f}") / 1e6)
    print(f, res[f], flush=True)
json.dump(res, open(f"{d}/onnx_check.json", "w"), indent=1)
if res["model_quantized.onnx"]["token_agreement"] < 0.995:
    print("int8 disagrees too much; removing model_quantized.onnx", flush=True)
    os.remove(f"{d}/onnx/model_quantized.onnx")
