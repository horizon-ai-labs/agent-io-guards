"""Web topic & format evaluation sets (evaluation only).   python webtax/build_evals.py OUT_DIR FW2_TEST.parquet
- en_topic.parquet / en_format.parquet: 3,000 random documents each from the test splits of WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8 and
  WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8 (DCLM web pages labelled by Llama-3.1-405B with the WebOrganizer prompts); column ref = label slug.
- fw2.parquet: the 9,750 held-out FineWeb-2 test documents of the edu classifier (65 languages; never trained on); labelled later by the
  independent judge (google/gemma-4-31b-it) with webtax/teacher_webtax.py.
"""
import importlib.util, os, sys
import pandas as pd
from huggingface_hub import HfApi, hf_hub_download
HERE = os.path.dirname(os.path.abspath(__file__))
sp = importlib.util.spec_from_file_location("t", os.path.join(HERE, "teacher_webtax.py")); m = importlib.util.module_from_spec(sp); sp.loader.exec_module(m)
OUT, FW = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True)
for t, repo in [("topics", "WebOrganizer/TopicAnnotations-Llama-3.1-405B-FP8"), ("formats", "WebOrganizer/FormatAnnotations-Llama-3.1-405B-FP8")]:
    f = [x for x in HfApi().list_repo_files(repo, repo_type="dataset") if "test" in x and x.endswith(".parquet")]
    d = pd.concat([pd.read_parquet(hf_hub_download(repo, x, repo_type="dataset"), columns=["text", "top_choice_index"]) for x in f])
    d = d.sample(3000, random_state=0); d["ref"] = [m.NAMES[t][i] for i in d.top_choice_index]; d["text"] = d.text.str[:6000]
    name = "en_topic" if t == "topics" else "en_format"; d[["text", "ref"]].reset_index(drop=True).to_parquet(f"{OUT}/{name}.parquet")
    print(name, len(d), d.ref.value_counts().head(5).to_dict())
fw = pd.read_parquet(FW)[["text", "lang"]]; fw.to_parquet(f"{OUT}/fw2.parquet"); print("fw2", len(fw))
