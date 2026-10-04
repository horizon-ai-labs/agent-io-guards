"""Print macro-F1 tables for fin/evals results.   python fin/table.py RESULTS.json [RESULTS.json ...]"""
import json, sys
import numpy as np
SETS = ["fpb_allagree", "fpb_50agree_hard", "tfns", "fiqa"] + ["mfs_" + l for l in ["en", "de", "fr", "es", "zh", "ja", "ar"]]
r = {}
for f in sys.argv[1:]:
    d = json.load(open(f))
    r.update({k: v for k, v in d.items() if isinstance(v, dict)} if "fpb_allagree" not in d else {f.split("/")[-1]: d})
print(" " * 24, " ".join(s.replace("fpb_", "").replace("50agree_hard", "50hard")[:8].rjust(8) for s in SETS), "  mfs_avg  en4_avg")
for m, v in r.items():
    name = m.split("/")[-2][:8] if "agent-outputs" in m else m.split("/")[-1][:24]
    f = lambda s: v[s]["macro_f1"] if s in v else float("nan")
    print(name.ljust(24), " ".join(f"{f(s):8.3f}" for s in SETS), f"{np.nanmean([f(s) for s in SETS[4:]]):8.3f}", f"{np.nanmean([f(s) for s in SETS[:4]]):8.3f}")
