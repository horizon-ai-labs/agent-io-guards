"""Held-out real hate-speech test sets added for the v2 gate (evaluation only; never trained on, never used for diagnosis before the gate).
python hate/build_evals2.py OUT_DIR
- hs18.parquet: Stormfront forum sentences (de Gibert et al. 2018; odegiber/hate_speech18, CC-BY-SA-3.0), English; hate vs noHate
  (idk/skip and relation dropped).
- ethos.parquet: ETHOS binary (github.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset, GPL-3.0), English YouTube/Reddit comments;
  hate = isHate >= 0.5.
- olidbr.parquet: OLID-BR test (dougtrajano/olid-br, CC-BY-4.0), Brazilian Portuguese; hate = racism, LGBTQphobia, religious intolerance,
  sexism or xenophobia; everything else (incl. other offensive comments) = 0.
"""
import io, os, sys, urllib.request, zipfile
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
z = zipfile.ZipFile(hf_hub_download("odegiber/hate_speech18", "data.zip", repo_type="dataset"))
meta = [n for n in z.namelist() if n.endswith("annotations_metadata.csv")][0]; root = meta.rsplit("/", 1)[0] + "/" if "/" in meta else ""
m = pd.read_csv(z.open(meta)); m = m[m.label.isin(["hate", "noHate"])]
m["text"] = [z.read(f"{root}all_files/{f}.txt").decode("utf-8", "ignore").strip() for f in m.file_id]
m["hate"] = (m.label == "hate").astype(int); m = m[m.text.str.len() > 0].drop_duplicates("text")
m[["text", "hate"]].reset_index(drop=True).to_parquet(f"{OUT}/hs18.parquet"); print("hs18", len(m), int(m.hate.sum()))
e = pd.read_csv(io.BytesIO(urllib.request.urlopen("https://raw.githubusercontent.com/intelligence-csd-auth-gr/Ethos-Hate-Speech-Dataset/master/ethos/ethos_data/Ethos_Dataset_Binary.csv").read()), sep=";")
e = e.rename(columns={"comment": "text"}); e["hate"] = (e.isHate >= 0.5).astype(int)
e[["text", "hate"]].drop_duplicates("text").reset_index(drop=True).to_parquet(f"{OUT}/ethos.parquet"); print("ethos", len(e), int(e.hate.sum()))
o = pd.read_parquet(hf_hub_download("dougtrajano/olid-br", [f for f in __import__("huggingface_hub").HfApi().list_repo_files("dougtrajano/olid-br", repo_type="dataset") if "test" in f and f.endswith(".parquet")][0], repo_type="dataset"))
H = ["racism", "lgbtqphobia", "religious_intolerance", "sexism", "xenophobia"]
o["hate"] = o[H].fillna(False).astype(bool).any(axis=1).astype(int)
o[["text", "hate"]].drop_duplicates("text").reset_index(drop=True).to_parquet(f"{OUT}/olidbr.parquet"); print("olidbr", len(o), int(o.hate.sum()))
