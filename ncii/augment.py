"""Add obfuscation-augmented rows to the NCII train set so the classifier is robust to homoglyphs, leetspeak, zero-width insertion,
character separation and small-caps (the normalizer only folds zero-width/invisibles; the rest must be learned, as in the injection guard).
Each augmented row keeps its label. Applied to a fraction of train rows (positives oversampled). Val/test stay clean.
python ncii/augment.py DATA_DIR FRACTION"""
import random, sys
import pandas as pd
d, frac = sys.argv[1], float(sys.argv[2]); rng = random.Random(0)
HOMO = {"a": "а", "e": "е", "o": "о", "c": "с", "p": "р", "x": "х", "y": "у", "i": "і", "s": "ѕ", "n": "ո"}
LEET = {"a": "4", "e": "3", "o": "0", "i": "1", "s": "5", "t": "7", "l": "1"}
SMALL = {c: s for c, s in zip("abcdefghijklmnopqrstuvwxyz", "ᴀʙᴄᴅᴇꜰɢʜɪᴊᴋʟᴍɴᴏᴩqʀsᴛᴜᴠᴡxyz")}


def homoglyph(t): return "".join(HOMO.get(c.lower(), c) if rng.random() < 0.6 else c for c in t)
def leet(t): return "".join(LEET.get(c.lower(), c) if rng.random() < 0.7 else c for c in t)
def zerow(t): return "​".join(t)
def charsep(t): return " ".join(t.replace(" ", ""))
def smallcaps(t): return "".join(SMALL.get(c.lower(), c) for c in t)
FNS = [homoglyph, leet, zerow, charsep, smallcaps]

tr = pd.read_parquet(f"{d}/train.parquet")
pool = pd.concat([tr[tr.label == 1], tr[tr.label == 1], tr[tr.label == 0]])  # oversample positives 2x
aug = pool.sample(frac=frac, random_state=0).copy()
aug["text"] = [rng.choice(FNS)(t) for t in aug.text]; aug["source"] = "aug_obf"; aug["kind"] = "aug_obf"
out = pd.concat([tr, aug]).sample(frac=1.0, random_state=0).reset_index(drop=True)
out.to_parquet(f"{d}/train.parquet", index=False)
print(f"train {len(tr)} + aug {len(aug)} = {len(out)}; aug pos {aug.label.mean():.2f}")
