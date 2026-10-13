"""Baseline: nvidia/prompt-task-and-complexity-classifier (inference code from its model card) on a prompt set.
python prompt/run_nvidia.py IN.parquet OUT.parquet   (adds nv_task, nv_creativity, nv_reasoning, nv_contextual_knowledge, nv_domain_knowledge,
nv_constraints, nv_complexity)"""
import sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from huggingface_hub import PyTorchModelHubMixin
from transformers import AutoConfig, AutoModel, AutoTokenizer


class MeanPooling(nn.Module):
    def forward(self, h, m):
        m = m.unsqueeze(-1).expand(h.size()).float(); return (h * m).sum(1) / m.sum(1).clamp(min=1e-9)


class MulticlassHead(nn.Module):
    def __init__(self, i, n):
        super().__init__(); self.fc = nn.Linear(i, n)

    def forward(self, x):
        return self.fc(x)


class CustomModel(nn.Module, PyTorchModelHubMixin):
    def __init__(self, target_sizes, task_type_map, weights_map, divisor_map):
        super().__init__()
        self.backbone = AutoModel.from_pretrained("microsoft/DeBERTa-v3-base"); self.target_sizes = target_sizes.values()
        self.task_type_map, self.weights_map, self.divisor_map = task_type_map, weights_map, divisor_map
        self.heads = [MulticlassHead(self.backbone.config.hidden_size, sz) for sz in self.target_sizes]
        for i, h in enumerate(self.heads):
            self.add_module(f"head_{i}", h)
        self.pool = MeanPooling()

    def forward(self, ids, att):
        h = self.backbone(input_ids=ids, attention_mask=att).last_hidden_state; p = self.pool(h, att)
        return [hd(p) for hd in self.heads]


import json, types
from huggingface_hub import hf_hub_download
cfg = types.SimpleNamespace(**json.load(open(hf_hub_download("nvidia/prompt-task-and-complexity-classifier", "config.json"))))  # no model_type -> AutoConfig fails on transformers 5
tok = AutoTokenizer.from_pretrained("nvidia/prompt-task-and-complexity-classifier")
m = CustomModel(target_sizes=cfg.target_sizes, task_type_map=cfg.task_type_map, weights_map=cfg.weights_map, divisor_map=cfg.divisor_map
                ).from_pretrained("nvidia/prompt-task-and-complexity-classifier").cuda().eval()
names = list(cfg.target_sizes.keys()); print(names, cfg.task_type_map, flush=True)
df = pd.read_parquet(sys.argv[1]); out = {k: [] for k in ["nv_task", "creativity_scope", "reasoning", "contextual_knowledge", "number_of_few_shots",
                                                         "domain_knowledge", "constraint_ct"]}
score = lambda lo, t: (torch.softmax(lo, 1).cpu().numpy() * np.array(cfg.weights_map[t])).sum(1) / cfg.divisor_map[t]
with torch.no_grad():
    for s in range(0, len(df), 64):
        e = tok(["Prompt: " + t for t in df.text.iloc[s:s + 64]], return_tensors="pt", max_length=512, padding=True, truncation=True)
        lo = m(e["input_ids"].cuda(), e["attention_mask"].cuda())
        out["nv_task"] += [cfg.task_type_map[str(i)] for i in lo[0].argmax(1).tolist()]
        for k, t in enumerate(names[1:], 1):
            if t in out:
                out[t] += list(score(lo[k].float(), t))
df["nv_task"] = out["nv_task"]
for a, b in [("creativity_scope", "nv_creativity"), ("reasoning", "nv_reasoning"), ("contextual_knowledge", "nv_contextual_knowledge"),
             ("domain_knowledge", "nv_domain_knowledge"), ("constraint_ct", "nv_constraints"), ("number_of_few_shots", "nv_few_shots")]:
    df[b] = out[a]
df["nv_complexity"] = (0.35 * df.nv_creativity + 0.25 * df.nv_reasoning + 0.15 * df.nv_constraints + 0.15 * df.nv_domain_knowledge
                       + 0.05 * df.nv_contextual_knowledge + 0.05 * df.nv_few_shots)
df.to_parquet(sys.argv[2]); print(df.nv_task.value_counts().to_dict(), flush=True)
