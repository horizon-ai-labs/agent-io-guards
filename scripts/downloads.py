"""Snapshot of Horizon-Labs model/dataset downloads (last-30-days counter shown on the Hub) and likes; diff vs the previous snapshot.
python scripts/downloads.py   -> appends to .gated/downloads.jsonl"""
import json, os, time
from huggingface_hub import HfApi
api = HfApi(); P = "/workspace/.gated/downloads.jsonl"
m = {x.id.split("/")[1]: (x.downloads or 0, x.likes or 0) for x in api.list_models(author="Horizon-Labs", expand=["downloads", "likes"])}
d = {x.id.split("/")[1]: (x.downloads or 0, x.likes or 0) for x in api.list_datasets(author="Horizon-Labs", expand=["downloads", "likes"])}
prev = json.loads(open(P).read().strip().split("\n")[-1]) if os.path.exists(P) else None
snap = dict(t=time.strftime("%Y-%m-%d %H:%M", time.gmtime()), models=m, datasets=d, total=sum(v[0] for v in m.values()))
open(P, "a").write(json.dumps(snap) + "\n")
print(snap["t"], "models total", snap["total"], f"(+{snap['total'] - prev['total']} since {prev['t']})" if prev else "")
for k, (dl, lk) in sorted(m.items(), key=lambda kv: -kv[1][0]):
    pd_ = prev["models"].get(k, [0, 0])[0] if prev else 0
    print(f"  {k:42s} {dl:6d} {'+' + str(dl - pd_) if prev else '':>6s} likes {lk}")
print("datasets", {k: v[0] for k, v in d.items()})
