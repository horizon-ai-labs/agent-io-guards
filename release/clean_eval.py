import json, re, sys
def clean(path, self_name=None):
    d = json.load(open(path)); out = {}
    for k, v in d.items():
        if "Llama-Prompt-Guard" in k: k = "meta-llama/" + k.rstrip("/").split("/")[-1]
        elif k.startswith("/"): k = self_name or "this-model"
        out[k] = v
    return out
