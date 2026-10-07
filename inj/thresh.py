"""Threshold calibration for an injection model from saved per-item predictions (train/evaluate.py preds/*.parquet).
python inj/thresh.py VAL_PREDS_DIR TEST_PREDS_DIR
T_ag = argmax F1 on val_agentic (ties -> higher threshold); T_g = argmax F1 on val_all. Prints card macro (11 external sets, as
inj/compare.py) and gate metrics at 0.5, T_g and T_ag."""
import glob, os, sys
import numpy as np, pandas as pd
EXT = [("notinject", "acc"), ("xstest", "acc"), ("orbench_hard", "acc"), ("qualifire", "f1"), ("jackhhao_test", "f1"), ("deepset_test", "f1"),
       ("simsonsun_jailbreaks", "acc"), ("boundary_pairs_test", "f1"), ("bipia", "f1"), ("piarena", "f1"), ("llmail_phase2", "acc")]
vd, td = sys.argv[1:3]
load = lambda d: {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in glob.glob(f"{d}/*.parquet")}
V, T = load(vd), load(td)


def met(df, t):
    y = df.label.values.astype(int); p = (df.score.values >= t).astype(int)
    tp = ((p == 1) & (y == 1)).sum(); fp = ((p == 1) & (y == 0)).sum(); fn = ((p == 0) & (y == 1)).sum()
    return dict(acc=float((p == y).mean()), f1=float(2 * tp / max(1, 2 * tp + fp + fn)), fpr=float(fp / max(1, (y == 0).sum())),
                recall=float(tp / max(1, (y == 1).sum())))


grid = np.round(np.arange(0.05, 0.996, 0.005), 3)
best = lambda df: max(grid, key=lambda t: (round(met(df, t)["f1"], 4), t))
T_ag, T_g = best(V["val_agentic"]), best(V["val_all"])
print(f"T_ag {T_ag} (val agentic F1 {met(V['val_agentic'], T_ag)['f1']:.3f} vs 0.5 {met(V['val_agentic'], 0.5)['f1']:.3f})")
print(f"T_g  {T_g} (val all F1 {met(V['val_all'], T_g)['f1']:.3f} vs 0.5 {met(V['val_all'], 0.5)['f1']:.3f})")
for name, t in [("0.5", 0.5), ("T_g", T_g), ("T_ag", T_ag)]:
    m = np.mean([met(T[s], t)[k] for s, k in EXT])
    g = {s: round(met(T[s], t)[k], 3) for s, k in [("mindgard_evasion", "acc"), ("simsonsun_jailbreaks", "acc"), ("boundary_pairs_test", "f1"), ("notinject", "acc")]}
    a = met(T["agentic5k_test"], t)
    print(f"{name:5s} t={t:.3f} macro {m:.4f} | {g} | agentic5k F1 {a['f1']:.3f} FPR {a['fpr']:.3f} recall {a['recall']:.3f}")
