"""bge-m3 (MIT) dense embeddings as distillation targets.   python emb/embed_teacher.py OUT_DIR SRC [SRC ...]
SRC = parquet path[:column] (default column "text"). Writes OUT_DIR/texts.parquet (text, src) and OUT_DIR/emb.npy
(float16, n x 1024, CLS pooling, L2-normalised, max 512 tokens) in the same order. Duplicated texts are embedded once.
"""
import os, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModel, AutoTokenizer

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
parts = []
for spec in sys.argv[2:]:
    path, col = (spec.rsplit(":", 1) if spec.endswith((":text", ":query")) else (spec, "text"))
    parts.append(pd.DataFrame({"text": pd.read_parquet(path, columns=[col])[col].astype(str), "src": os.path.basename(path) + ":" + col}))
df = pd.concat(parts, ignore_index=True).drop_duplicates("text").reset_index(drop=True)
sh, ns = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))   # optional sharding (rows sh::ns)
df = df.iloc[sh::ns].reset_index(drop=True)
print("texts", len(df), df.src.value_counts().to_dict(), flush=True)
tok = AutoTokenizer.from_pretrained("BAAI/bge-m3"); m = AutoModel.from_pretrained("BAAI/bge-m3", torch_dtype=torch.float16).cuda().eval()
E = np.lib.format.open_memmap(f"{out}/emb.npy", mode="w+", dtype=np.float16, shape=(len(df), 1024))
texts = df.text.tolist(); order = np.argsort([len(t) for t in texts]); bs = 256
with torch.no_grad():
    for s in range(0, len(texts), bs):
        b = order[s:s + bs]
        e = tok([texts[i] for i in b], truncation=True, max_length=512, padding=True, return_tensors="pt").to("cuda")
        h = m(**e).last_hidden_state[:, 0]
        E[np.sort(b)] = torch.nn.functional.normalize(h.float(), dim=-1).half().cpu().numpy()[np.argsort(b)]
        if s % (bs * 400) == 0:
            print(s, flush=True)
E.flush(); df.to_parquet(f"{out}/texts.parquet"); print("done", flush=True)
