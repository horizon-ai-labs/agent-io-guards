"""MultiCoNER v2 train (CC-BY-4.0) mapped to PER/ORG/LOC/MISC, in pii/train_tok.py format.   python ner/build_multiconer.py OUT.parquet [PER_LANG]
Person types -> PER; groups/companies/teams -> ORG; facilities/settlements/stations/other locations -> LOC; creative works and
products -> MISC; medical types -> O. Tokens joined with spaces (no spaces for Chinese). Text is lowercase in the source.
"""
import json, random, sys
import pandas as pd
from huggingface_hub import hf_hub_download
OUT = sys.argv[1]; PER = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
M = {**{t: "PER" for t in ["Artist", "Athlete", "Politician", "Cleric", "Scientist", "SportsManager", "OtherPER"]},
     **{t: "ORG" for t in ["MusicalGRP", "PublicCorp", "PrivateCorp", "AerospaceManufacturer", "SportsGRP", "CarManufacturer", "TechCORP", "ORG"]},
     **{t: "LOC" for t in ["Facility", "OtherLOC", "HumanSettlement", "Station"]},
     **{t: "MISC" for t in ["VisualWork", "MusicalWork", "WrittenWork", "ArtWork", "Software", "Clothing", "Vehicle", "Food", "Drink", "OtherPROD"]}}
rows = []
for d, lang in [("BN-Bangla", "bn"), ("DE-German", "de"), ("EN-English", "en"), ("ES-Spanish", "es"), ("FA-Farsi", "fa"), ("FR-French", "fr"),
                ("HI-Hindi", "hi"), ("IT-Italian", "it"), ("PT-Portuguese", "pt"), ("SV-Swedish", "sv"), ("UK-Ukrainian", "uk"), ("ZH-Chinese", "zh")]:
    sents, cur = [], []
    for line in open(hf_hub_download("MultiCoNER/multiconer_v2", f"{d}/{lang}_train.conll", repo_type="dataset"), encoding="utf-8"):
        line = line.rstrip("\n")
        if line.startswith("# id") or not line.strip():
            if cur:
                sents.append(cur); cur = []
            continue
        parts = line.split(" ")
        if len(parts) >= 4:
            cur.append((parts[0], parts[-1]))
    if cur:
        sents.append(cur)
    random.Random(0).shuffle(sents)
    for s in sents[:PER]:
        text, spans, ent = "", [], None
        for w, tag in s:
            if text and lang != "zh":
                text += " "
            a = len(text); text += w; b = len(text)
            t = tag[2:] if tag != "O" else None; ty = M.get(t) if t else None
            if tag.startswith("B-") or ty is None or (ent and ent[2] != ty):
                if ent:
                    spans.append(ent); ent = None
            if ty is not None:
                ent = [ent[0], b, ty] if ent and not tag.startswith("B-") else [a, b, ty]
        if ent:
            spans.append(ent)
        rows.append((text, json.dumps(spans), f"multiconer_{lang}"))
df = pd.DataFrame(rows, columns=["text", "spans_json", "source"]); df.to_parquet(OUT)
print(len(df), df.source.value_counts().to_dict())
