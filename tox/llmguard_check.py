"""Check that multilingual-toxicity works as a drop-in model for LLM Guard's Toxicity scanners (input and output) - its labels are the
Detoxify set the scanner expects. Private venv with llm-guard. python tox/llmguard_check.py MODEL_DIR"""
import json, sys, importlib.metadata as md
M = sys.argv[1]
from llm_guard.model import Model
from llm_guard.input_scanners import Toxicity
from llm_guard.output_scanners import Toxicity as OutToxicity
T = [("You are a worthless idiot and everyone hates you.", True), ("Thanks for the help, have a great weekend!", False),
     ("Du bist ein dummer Vollidiot.", True), ("Wie spät ist es in Berlin?", False),
     ("Eres un imbécil y nadie te soporta.", True), ("¿Me recomiendas un buen libro?", False),
     ("Ты тупой идиот.", True), ("Какая погода завтра в Москве?", False)]
kw = {"top_k": None, "function_to_apply": "sigmoid", "return_token_type_ids": False, "max_length": 512, "truncation": True}
res = {"versions": {p: md.version(p) for p in ["llm-guard", "transformers", "torch", "onnxruntime", "optimum"] if p in {d.metadata["Name"].lower() for d in md.distributions()}}}
for use_onnx in (True, False):
    model = Model(path=M, onnx_path=M, onnx_subfolder="onnx", pipeline_kwargs=kw)
    sc, osc = Toxicity(model=model, threshold=0.5, use_onnx=use_onnx), OutToxicity(model=model, threshold=0.5, use_onnx=use_onnx)
    rows = []
    for t, toxic in T:
        _, valid, risk = sc.scan(t); _, ovalid, _ = osc.scan("prompt", t)
        rows.append(dict(text=t[:40], toxic=toxic, input_flagged=not valid, output_flagged=not ovalid, correct=(not valid) == toxic, risk=round(float(risk), 3)))
    res["onnx" if use_onnx else "torch"] = dict(all_correct=all(r["correct"] for r in rows), rows=rows)
print(json.dumps(res, indent=1, ensure_ascii=False))
