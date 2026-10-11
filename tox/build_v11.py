"""v1.1 training data: v1.0 data (tox/build_tox.py) + native comments (tox/gen_native.py -> tox/build_native.py) with the 7
Qwen3.8-27B teacher labels (tox/teacher_tox.py label). Validation set unchanged (v1.0 val).
python tox/build_v11.py V0_DIR OUT_DIR LABELLED_SHARD.parquet..."""
import json, os, shutil, sys
import pandas as pd
LABELS = ["toxicity", "severe_toxicity", "obscene", "threat", "insult", "identity_attack", "sexual_explicit"]
v0, out = sys.argv[1], sys.argv[2]
nat = pd.concat([pd.read_parquet(p) for p in sys.argv[3:]], ignore_index=True)
tr = pd.read_parquet(f"{v0}/train.parquet")
nat = nat[~nat.text.isin(set(tr.text))].drop_duplicates("text")
nat = nat.assign(src_idx=-1)[["text", "lang", "src_idx"] + LABELS]
df = pd.concat([tr, nat.astype(tr.dtypes.to_dict())], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True)
os.makedirs(out, exist_ok=True); df.to_parquet(f"{out}/train.parquet"); shutil.copy(f"{v0}/val.parquet", f"{out}/val.parquet")
st = json.load(open(f"{v0}/stats.json")); st["v11_native"] = dict(n=len(nat), mean_labels={l: round(float(nat[l].mean()), 4) for l in LABELS},
                                                                   frac_toxic_ge_05=round(float((nat.toxicity >= 0.5).mean()), 4))
st["n_train_v11"] = len(df); json.dump(st, open(f"{out}/stats.json", "w"), indent=1); print(json.dumps(st["v11_native"]), len(df))
