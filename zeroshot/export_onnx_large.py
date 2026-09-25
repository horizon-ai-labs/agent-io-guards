"""ONNX export for >2 GB models (bge-m3 / XLM-R large): fp32 with external data + dynamic int8 (all MatMuls; XLM-R
has no mmBERT-style activation outliers, verified by zeroshot/requant_check.sh-style top-label agreement).
python zeroshot/export_onnx_large.py MODEL_DIR -> MODEL_DIR/onnx/{model.onnx, model.onnx_data, model_quantized.onnx}"""
import os, sys
import numpy as np, onnx, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer
from onnxruntime.quantization import quantize_dynamic, QuantType

d = sys.argv[1]; os.makedirs(f"{d}/onnx", exist_ok=True)
tok = AutoTokenizer.from_pretrained(d)
model = AutoModelForSequenceClassification.from_pretrained(d, attn_implementation="eager").eval()
enc = tok(["The new phone's battery lasts all day.", "Mañana juega el Real Madrid."], ["This text is about technology.", "This text is about sports."],
          return_tensors="pt", padding=True)


class Wrap(torch.nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, input_ids, attention_mask):
        return self.m(input_ids=input_ids, attention_mask=attention_mask).logits


tmp = f"{d}/onnx/tmp"; os.makedirs(tmp, exist_ok=True)
with torch.no_grad():
    torch.onnx.export(Wrap(model), (enc["input_ids"], enc["attention_mask"]), f"{tmp}/model.onnx",
                      input_names=["input_ids", "attention_mask"], output_names=["logits"],
                      dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"}, "logits": {0: "batch"}},
                      opset_version=17, dynamo=False)
m = onnx.load(f"{tmp}/model.onnx")   # re-save with a single external data file
onnx.save_model(m, f"{d}/onnx/model.onnx", save_as_external_data=True, all_tensors_to_one_file=True, location="model.onnx_data")
del m
quantize_dynamic(f"{d}/onnx/model.onnx", f"{d}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8, use_external_data_format=False)
import shutil; shutil.rmtree(tmp)
for f in os.listdir(f"{d}/onnx"):
    print(f, round(os.path.getsize(f"{d}/onnx/{f}") / 1e6), "MB")
