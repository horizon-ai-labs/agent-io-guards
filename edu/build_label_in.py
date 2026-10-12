"""Teacher-labelling input for the educational-quality classifier:
- FineWeb-2 documents (edu/prep_fw.py train_docs), PER per language (English FineWeb docs excluded: English comes from Llama-3 annotations);
- 10,000 documents from shard 0 of HuggingFaceFW/fineweb-edu-llama3-annotations with their Llama-3 score ('llama'), to fit the
  teacher -> Llama-3 calibration (edu/calibrate.py).
python edu/build_label_in.py TRAIN_DOCS.parquet OUT.parquet [PER=2000]"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download
d = pd.read_parquet(sys.argv[1]); d = d[d.lang != "eng_Latn"]
d = d.sample(frac=1, random_state=0).groupby("lang").head(int(os.environ.get("PER", 2000))).assign(source="fineweb2")
l = pd.read_parquet(hf_hub_download("HuggingFaceFW/fineweb-edu-llama3-annotations", "data/train-00000-of-00008.parquet", repo_type="dataset"), columns=["text", "score"])
l = l.sample(10000, random_state=0).rename(columns={"score": "llama"}).assign(lang="eng_Latn", source="llama_calib"); l["text"] = l.text.str[:6000]
o = pd.concat([d, l], ignore_index=True).sample(frac=1, random_state=0).reset_index(drop=True); o.to_parquet(sys.argv[2])
print(len(o), o.source.value_counts().to_dict(), o.lang.nunique())
