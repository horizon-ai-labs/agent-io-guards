"""Package an embedding student as a sentence-transformers model + ONNX (sentence_embedding output) + int8 check.
python emb/build_release.py SRC_DIR BASE_HUB_ID OUT_DIR
SRC_DIR: emb/train_emb.py output (encoder + projection.pt). OUT_DIR: Transformer weights at the root, 1_Pooling (mean),
2_Dense (hidden -> 1024, no bias), 3_Normalize; onnx/model.onnx + onnx/model_quantized.onnx (inputs input_ids,
attention_mask; output sentence_embedding = normalised mean-pooled projection) and onnx_sweep.json (cosine fp32 vs int8).
"""
import json, os, shutil, sys
import numpy as np, torch
from huggingface_hub import hf_hub_download
from safetensors.torch import save_file
from transformers import AutoModel, AutoTokenizer

src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "val_metrics.json", "train_log.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
P = torch.load(f"{src}/projection.pt", map_location="cpu").float()   # (hidden, 1024)
hid = P.shape[0]
json.dump([{"idx": 0, "name": "0", "path": "", "type": "sentence_transformers.models.Transformer"},
           {"idx": 1, "name": "1", "path": "1_Pooling", "type": "sentence_transformers.models.Pooling"},
           {"idx": 2, "name": "2", "path": "2_Dense", "type": "sentence_transformers.models.Dense"},
           {"idx": 3, "name": "3", "path": "3_Normalize", "type": "sentence_transformers.models.Normalize"}], open(f"{out}/modules.json", "w"), indent=1)
json.dump({"max_seq_length": 512, "do_lower_case": False}, open(f"{out}/sentence_bert_config.json", "w"))
json.dump({"similarity_fn_name": "cosine", "prompts": {}, "default_prompt_name": None}, open(f"{out}/config_sentence_transformers.json", "w"))
os.makedirs(f"{out}/1_Pooling", exist_ok=True); os.makedirs(f"{out}/2_Dense", exist_ok=True); os.makedirs(f"{out}/3_Normalize", exist_ok=True)
json.dump({"word_embedding_dimension": hid, "pooling_mode_cls_token": False, "pooling_mode_mean_tokens": True, "pooling_mode_max_tokens": False,
           "pooling_mode_mean_sqrt_len_tokens": False, "pooling_mode_weightedmean_tokens": False, "pooling_mode_lasttoken": False,
           "include_prompt": True}, open(f"{out}/1_Pooling/config.json", "w"), indent=1)
json.dump({"in_features": hid, "out_features": 1024, "bias": False, "activation_function": "torch.nn.modules.linear.Identity"},
          open(f"{out}/2_Dense/config.json", "w"), indent=1)
save_file({"linear.weight": P.T.contiguous()}, f"{out}/2_Dense/model.safetensors")

tok = AutoTokenizer.from_pretrained(out); enc_m = AutoModel.from_pretrained(out, attn_implementation="eager").eval()


class Wrap(torch.nn.Module):
    def __init__(self, m, P):
        super().__init__(); self.m = m; self.P = torch.nn.Parameter(P, requires_grad=False)

    def forward(self, input_ids, attention_mask):
        h = self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        a = attention_mask.unsqueeze(-1).to(h.dtype); v = (h * a).sum(1) / a.sum(1)
        return torch.nn.functional.normalize(v @ self.P, dim=-1)


texts = ["How tall is the Eiffel Tower?", "La tour Eiffel mesure 330 mètres.", "电动汽车电池能用多久", "Der Eiffelturm ist mit 330 Metern das höchste Bauwerk von Paris. " * 8]
e = tok(texts, return_tensors="pt", padding=True)
os.makedirs(f"{out}/onnx", exist_ok=True); w = Wrap(enc_m, P).eval()
with torch.no_grad():
    ref = w(e["input_ids"], e["attention_mask"]).numpy()
    torch.onnx.export(w, (e["input_ids"], e["attention_mask"]), f"{out}/onnx/model.onnx", input_names=["input_ids", "attention_mask"],
                      output_names=["sentence_embedding"], dynamic_axes={"input_ids": {0: "batch", 1: "seq"}, "attention_mask": {0: "batch", 1: "seq"},
                                                                          "sentence_embedding": {0: "batch"}}, opset_version=17, dynamo=False)
import onnxruntime as ort
from onnxruntime.quantization import quantize_dynamic, QuantType
quantize_dynamic(f"{out}/onnx/model.onnx", f"{out}/onnx/model_quantized.onnx", weight_type=QuantType.QInt8, op_types_to_quantize=["Gather"])
feeds = {k: e[k].numpy() for k in ["input_ids", "attention_mask"]}
r = {}
for f in ["model.onnx", "model_quantized.onnx"]:
    o = ort.InferenceSession(f"{out}/onnx/{f}", providers=["CPUExecutionProvider"]).run(None, feeds)[0]
    r[f] = dict(cos_to_torch=float((o * ref).sum(-1).min()), mb=os.path.getsize(f"{out}/onnx/{f}") / 1e6)
json.dump(dict(gather_only=dict(min_cos=r["model_quantized.onnx"]["cos_to_torch"], mb=r["model_quantized.onnx"]["mb"]), fp32=r["model.onnx"], n=len(texts)),
          open(f"{out}/onnx_sweep.json", "w"), indent=1)
print(r, flush=True)
