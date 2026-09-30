"""Language-ID data v1 = v0 with (1) Akan/Twi merged, (2) English contamination removed.   python lid/clean_lid.py MODEL V0_DIR OUT

(1) FLORES aka_Latn had no FineWeb-2 subset and was aliased to twi_Latn, so both labels had the same training text; Twi is a
    variety of Akan -> one label aka_Latn (train and eval; FLORES twi_Latn items count as aka_Latn).
(2) v0 scored English only .87 on FLORES and predicted English sentences as ban_Latn, afr_Latn but also dzo_Tibt, lao_Laoo,
    mni_Beng: FineWeb-2 subsets contain English (and other off-script) pages. Filters (env FILTERS, default all):
    - model: label != eng_Latn and v0 P(eng_Latn) > 0.9 (caught only 19 samples in v1, because v0 had fitted them)
    - script: for non-Latin-script labels, drop samples in which < 50% of the letters are in the label's script
    - english: for Latin-script labels other than eng_Latn, drop samples whose word tokens are >= 25% common English
      function words (the, and, of, to, is, ...) with at least 4 words
    Also reports what FLORES English sentences are predicted as with v0.
"""
import json, os, shutil, sys
import numpy as np, pandas as pd, torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

md, v0, out = sys.argv[1:4]
os.makedirs(f"{out}/eval", exist_ok=True)
tok = AutoTokenizer.from_pretrained(md); m = AutoModelForSequenceClassification.from_pretrained(md, torch_dtype=torch.bfloat16).cuda().eval()
id2 = m.config.id2label; ENG = m.config.label2id["eng_Latn"]


@torch.no_grad()
def probs_eng_top(texts, bs=512):
    pe, top, tp = np.zeros(len(texts)), [None] * len(texts), np.zeros(len(texts))
    order = np.argsort([len(t) for t in texts])
    for s in range(0, len(texts), bs):
        b = order[s:s + bs]
        enc = tok([texts[i] for i in b], truncation=True, max_length=128, padding=True, return_tensors="pt")
        p = torch.softmax(m(input_ids=enc["input_ids"].cuda(), attention_mask=enc["attention_mask"].cuda()).logits.float(), -1)
        pe[b] = p[:, ENG].cpu().numpy(); v, i = p.max(-1)
        tp[b] = v.cpu().numpy()
        for k, j in zip(b, i.tolist()):
            top[k] = id2[j]
    return pe, top, tp


merge = lambda l: "aka_Latn" if l == "twi_Latn" else l
import re, unicodedata
SCRIPT = dict(Latn=("LATIN",), Cyrl=("CYRILLIC",), Arab=("ARABIC",), Deva=("DEVANAGARI",), Beng=("BENGALI",), Tibt=("TIBETAN",),
              Laoo=("LAO",), Thai=("THAI",), Grek=("GREEK",), Hebr=("HEBREW",), Ethi=("ETHIOPIC",), Hang=("HANGUL",), Kore=("HANGUL",),
              Jpan=("HIRAGANA", "KATAKANA", "CJK"), Hani=("CJK",), Hans=("CJK",), Hant=("CJK",), Armn=("ARMENIAN",), Geor=("GEORGIAN",),
              Gujr=("GUJARATI",), Guru=("GURMUKHI",), Knda=("KANNADA",), Mlym=("MALAYALAM",), Orya=("ORIYA",), Sinh=("SINHALA",),
              Taml=("TAMIL",), Telu=("TELUGU",), Mymr=("MYANMAR",), Khmr=("KHMER",), Tfng=("TIFINAGH",), Olck=("OL CHIKI",))
EN = set("the and of to is in that for with on are was this as by be it from at or an have not which you".split())
FILTERS = set(os.environ.get("FILTERS", "model,script,english").split(","))


def script_frac(t, sc):
    pre = SCRIPT.get(sc)
    if pre is None:
        return 1.0
    letters = [c for c in t if c.isalpha()]
    if not letters:
        return 1.0
    return sum(unicodedata.name(c, "").startswith(pre) for c in letters) / len(letters)


def english_like(t):
    w = re.findall(r"[a-z]+", t.lower())
    return len(w) >= 4 and sum(x in EN for x in w) / len(w) >= 0.25
ev = pd.read_parquet(f"{v0}/eval/flores_devtest.parquet")
eng = ev[ev.label == "eng_Latn"]
_, top, _ = probs_eng_top(eng.text.tolist())
print("FLORES English predicted as:", pd.Series(top).value_counts().head(10).to_dict(), flush=True)
report = dict(flores_eng_pred=pd.Series(top).value_counts().head(20).to_dict())
for split in ["train", "val"]:
    df = pd.read_parquet(f"{v0}/{split}.parquet")
    pe, _, _ = probs_eng_top(df.text.tolist())
    sc = df.label.str.split("_").str[-1].values
    d_model = (df.label != "eng_Latn").values & (pe > 0.9) if "model" in FILTERS else np.zeros(len(df), bool)
    d_script = np.array([s != "Latn" and script_frac(t, s) < 0.5 for t, s in zip(df.text, sc)]) if "script" in FILTERS else np.zeros(len(df), bool)
    d_eng = np.array([s == "Latn" and l != "eng_Latn" and english_like(t) for t, s, l in zip(df.text, sc, df.label)]) if "english" in FILTERS else np.zeros(len(df), bool)
    drop = d_model | d_script | d_eng
    report[f"{split}_dropped_filters"] = dict(model=int(d_model.sum()), script=int(d_script.sum()), english=int(d_eng.sum()))
    print(split, report[f"{split}_dropped_filters"], flush=True)
    report[f"{split}_dropped"] = int(drop.sum())
    report[f"{split}_dropped_by_label"] = df[drop].label.value_counts().head(30).to_dict()
    df = df[~drop].assign(label=lambda d: d.label.map(merge)).drop_duplicates(subset=["text"])
    df.to_parquet(f"{out}/{split}.parquet")
    print(split, "dropped", int(drop.sum()), "of", len(drop), dict(list(report[f"{split}_dropped_by_label"].items())[:12]), flush=True)
for f in ["flores_devtest", "flores_short"]:
    pd.read_parquet(f"{v0}/eval/{f}.parquet").assign(label=lambda d: d.label.map(merge)).to_parquet(f"{out}/eval/{f}.parquet")
labels = sorted({merge(l) for l in json.load(open(f"{v0}/labels.json"))})
json.dump(labels, open(f"{out}/labels.json", "w")); json.dump(report, open(f"{out}/clean_report.json", "w"), indent=1)
shutil.copy(f"{v0}/stats.json", f"{out}/stats_v0.json")
print("labels", len(labels))
