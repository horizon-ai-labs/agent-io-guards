"""Language-ID data v5 = v4 with unreliable fine-grained labels merged or dropped.   python lid/merge_labels.py V4_DIR OUT
Why (v3/v4 evidence): Arabic dialect labels were near-useless on FLORES (ars .01, acm .15, apc .33, ary .40, aeb .65, arz .76)
and pulled Modern Standard Arabic away from arb_Arab (4 of 4 hand-written MSA phrases labelled as dialects); FineWeb-2's
dialect subsets contain much MSA. Cantonese (yue_Hant, FLORES acc .03, GlotLID .00) took Mandarin phrases ("早上好" -> yue).
Dyula (dyu, .03) and Bambara are mutually intelligible Manding varieties.
Merges: ars/acm/apc/ary/aeb/arz_Arab -> arb_Arab ("Arabic"); dyu_Latn -> bam_Latn. Dropped: yue_Hant (training rows removed,
listed as unsupported). The same map is applied to every model's predictions in lid/eval_lid.py.
"""
import json, os, shutil, sys
import pandas as pd

V, OUT = sys.argv[1:3]
MERGE = {k: "arb_Arab" for k in ["ars_Arab", "acm_Arab", "apc_Arab", "ary_Arab", "aeb_Arab", "arz_Arab"]}
MERGE["dyu_Latn"] = "bam_Latn"
DROP = {"yue_Hant"}
os.makedirs(f"{OUT}/eval", exist_ok=True)
for split in ["train", "val"]:
    df = pd.read_parquet(f"{V}/{split}.parquet")
    df = df[~df.label.isin(DROP)].assign(label=lambda d: d.label.replace(MERGE))
    df.to_parquet(f"{OUT}/{split}.parquet"); print(split, len(df))
for f in ["flores_devtest", "flores_short"]:
    e = pd.read_parquet(f"{V}/eval/{f}.parquet")
    e[~e.label.isin(DROP)].assign(label=lambda d: d.label.replace(MERGE)).to_parquet(f"{OUT}/eval/{f}.parquet")
labels = sorted({MERGE.get(l, l) for l in json.load(open(f"{V}/labels.json")) if l not in DROP})
json.dump(labels, open(f"{OUT}/labels.json", "w")); json.dump(dict(merge=MERGE, drop=sorted(DROP)), open(f"{OUT}/merges.json", "w"))
print("labels", len(labels))
