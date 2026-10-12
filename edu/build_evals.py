"""Educational-quality evaluation sets.   python edu/build_evals.py OUT_DIR
- fw2_gemma.parquet: 9,750 held-out FineWeb-2 documents (65 languages, 150 each; edu/prep_fw.py test split, never trained on) scored with the
  same rubric by an independent judge, google/gemma-4-31b-it (expected score; edu/run_judge.sh). Built from the judge output, not by this script.
- en_llama3.parquet: 6,000 random documents from shard 7 of HuggingFaceFW/fineweb-edu-llama3-annotations (ODC-BY; English web pages scored
  0-5 by Llama-3-70B-Instruct with the FineWeb-Edu prompt). Shard 7 is never used for training. Columns text, score.
"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download
OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
d = pd.read_parquet(hf_hub_download("HuggingFaceFW/fineweb-edu-llama3-annotations", "data/train-00007-of-00008.parquet", repo_type="dataset"), columns=["text", "score"])
d = d.sample(6000, random_state=0).reset_index(drop=True); d.to_parquet(f"{OUT}/en_llama3.parquet"); print("en_llama3", len(d), d.score.value_counts().sort_index().to_dict())
