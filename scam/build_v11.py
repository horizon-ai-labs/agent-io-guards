"""Scam-detector data v1.1 = data v0 (unchanged) + teacher-labelled real-style notifications (scam/gen_notif.py) + accent-free copies.
Accent-free copies: real SMS is often sent without diacritics (GSM-7 charset). For 30% of SMS / push texts in Latin-script languages whose
folded form differs, add a folded copy with the same teacher probabilities (label-preserving), from both v0 and the new notifications.
Texts in scam/evals or scam/evals_bulwark are removed (normalised exact match). Val: v0 val + 3% of the new rows (and their copies).
python scam/build_v11.py OUT V0_DIR LABELLED.parquet [...]"""
import glob, json, os, re, sys, unicodedata
import numpy as np, pandas as pd

OUT, V0, ins = sys.argv[1], sys.argv[2], sys.argv[3:]; os.makedirs(OUT, exist_ok=True)
H = os.path.dirname(os.path.abspath(__file__))
norm = lambda s: re.sub(r"\W+", " ", s.lower()).strip()
ev = {norm(t) for d in ("evals", "evals_bulwark") for p in glob.glob(f"{H}/{d}/*.parquet") for t in pd.read_parquet(p).text}
EXTRA = str.maketrans({"ı": "i", "İ": "I", "ß": "ss", "ø": "o", "Ø": "O", "æ": "ae", "Æ": "AE", "ł": "l", "Ł": "L", "đ": "d", "Đ": "D",
                       "ð": "d", "þ": "th", "œ": "oe", "Œ": "OE"})


def fold(t):
    t = unicodedata.normalize("NFKD", t.translate(EXTRA))
    return unicodedata.normalize("NFC", "".join(c for c in t if not unicodedata.combining(c)))


def latin_share(t):
    L = [c for c in t if c.isalpha()]
    return sum("LATIN" in unicodedata.name(c, "") for c in L) / max(1, len(L))


tr, va = pd.read_parquet(f"{V0}/train.parquet"), pd.read_parquet(f"{V0}/val.parquet")
new = pd.concat([pd.read_parquet(p) for p in ins], ignore_index=True)
st = dict(v0_train=len(tr), v0_val=len(va), new_in=len(new))
new = new[new.text.str.len() >= 5].drop_duplicates("text")
new = new[~new.text.map(norm).isin(ev) & ~new.text.isin(set(tr.text) | set(va.text))]
new = new.sample(frac=1.0, random_state=0).reset_index(drop=True); nv = int(0.03 * len(new))
tr, va = pd.concat([tr, new.iloc[nv:]], ignore_index=True), pd.concat([va, new.iloc[:nv]], ignore_index=True)


def copies(df, seed):
    rng = np.random.default_rng(seed)
    c = df[df.channel.fillna("").str.contains("SMS|push") & (rng.random(len(df)) < 0.3)].copy()
    c = c[c.text.map(latin_share) >= 0.8]
    c["text"] = c.text.map(fold); c = c[c.text.values != df.loc[c.index, "text"].values]
    c["src"] = c.src + "_fold"; return c


ct, cv = copies(tr, 1), copies(va, 2)
tr = pd.concat([tr, ct], ignore_index=True).drop_duplicates("text").sample(frac=1.0, random_state=1).reset_index(drop=True)
va = pd.concat([va, cv], ignore_index=True).drop_duplicates("text").reset_index(drop=True)
tr.to_parquet(f"{OUT}/train.parquet"); va.to_parquet(f"{OUT}/val.parquet")
arg = lambda d: np.array(["legitimate", "spam", "fraud"])[d[["p_legit", "p_spam", "p_fraud"]].values.argmax(1)]
st.update(new_kept=len(new), fold_train=len(ct), fold_val=len(cv), n_train=len(tr), n_val=len(va),
          label_by_src=pd.crosstab(tr.src, arg(tr)).to_dict("index"),
          notif_intended_vs_teacher=pd.crosstab(new.intended, arg(new)).to_dict("index"))
json.dump(st, open(f"{OUT}/stats.json", "w"), indent=1, default=int); print(json.dumps(st, default=int, indent=1))
