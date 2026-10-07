"""Injection data v2.4 and v2.5 (the data of small v2.3), reproducing the steps run on 2026-10-05/06:
v2.4 = v2.3b (inj/build_v23.py output with teacher soft labels for the hijack pairs) + harmless trigger-word commands
       (inj/gen_benign_cmd.py; label 0; soft = the Qwen judge's score, NOT max(large, qwen): the large v2.1 model over-flags exactly
       these commands); 97% train / 3% val.
v2.5 = v2.4 train + one extra copy of the jailbreak-style sources (gen2_framing/* both labels, nvidia_jailbreak, jailbreakv/* positives).
python inj/build_v24_v25.py V23B_DIR BENIGN_CMD_WITH_QWEN.parquet OUT_V24 OUT_V25
(BENIGN_CMD_WITH_QWEN: text, label, source, kind, lang, qwen, soft_large; from inj/teacher_inj.py label mode and train/evaluate.py preds)"""
import os, sys
import pandas as pd
v23b, cmd, o24, o25 = sys.argv[1:5]
n = pd.read_parquet(cmd); n["soft"] = n["qwen"]
part = n.sample(frac=1.0, random_state=0); k = int(len(part) * 0.97)
for o in (o24, o25):
    os.makedirs(o, exist_ok=True)
for sp, chunk in [("train", part.iloc[:k]), ("val", part.iloc[k:])]:
    base = pd.read_parquet(f"{v23b}/{sp}.parquet")
    df = pd.concat([base, chunk[["text", "label", "source", "kind", "lang", "soft", "qwen", "soft_large"]]], ignore_index=True).sample(frac=1.0, random_state=0)
    df.to_parquet(f"{o24}/{sp}.parquet"); print("v2.4", sp, len(df))
d = pd.read_parquet(f"{o24}/train.parquet")
m = d.source.str.startswith("gen2_framing/") | d.source.isin(["nvidia_jailbreak"]) | (d.source.str.startswith("jailbreakv/") & (d.label == 1))
pd.concat([d, d[m]], ignore_index=True).sample(frac=1.0, random_state=0).to_parquet(f"{o25}/train.parquet")
pd.read_parquet(f"{o24}/val.parquet").to_parquet(f"{o25}/val.parquet"); print("v2.5 extra rows", int(m.sum()))
