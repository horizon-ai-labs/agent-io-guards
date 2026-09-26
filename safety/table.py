"""Print a comparison table from eval_safety.py outputs.  python safety/table.py a.json b.json ... [--metric f1]"""
import json, sys
metric = sys.argv[sys.argv.index("--metric") + 1] if "--metric" in sys.argv else "f1"
files = [a for i, a in enumerate(sys.argv[1:], 1) if not a.startswith("--") and sys.argv[i - 1] != "--metric"]
res = {}
for f in files:
    for k, v in json.load(open(f)).items():
        name = k.split("/")[-2][:8] if k.startswith("/fast") else k.split("/")[-1]
        res[name] = v
evals = sorted({e for v in res.values() for e in v if not e.startswith(("_", "categories_"))})
w = max(len(n) for n in res)
print(" " * w, " ".join(f"{e[:11]:>11s}" for e in evals))
for n, v in res.items():
    cells = []
    for e in evals:
        m = v.get(e, {}); x = m.get(metric) if m else None
        if x is None and metric == "f1" and m and m.get("recall") is not None and m.get("fpr") is None:
            x = m["recall"]   # all-unsafe sets: show recall
        cells.append(f"{x:11.3f}" if x is not None else f"{'-':>11s}")
    print(f"{n:{w}s}", " ".join(cells))
