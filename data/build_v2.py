"""Data v2 = data v1 + gen v2 (framing pairs, task-hijack docs) + character-level evasion augmentation.

env: V1=<dir with v1 train/val/eval>  GEN=<glob of gen2_*.jsonl>  OUT=<dir>  DROP=<regex of v1 sources to drop>
The augmentation perturbs attacks AND benign texts with the same operators, so unusual characters are not a
shortcut for "injection". (Mindgard's eval set uses similar perturbation families; the card says so.)
"""
import glob, json, os, random, re, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
V1, OUT = os.environ["V1"], os.environ["OUT"]
GEN = sorted(glob.glob(os.environ["GEN"]))
DROP = os.environ.get("DROP", "")
os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(11)
from langs import LANG  # noqa: E402


def norm(t):
    return re.sub(r"\W+", " ", t.lower()).strip()


def rec(text, label, source, kind="direct", lang="en"):
    return dict(text=str(text).strip(), label=int(label), source=source, kind=kind, lang=lang)


# ---------------------------------------------------------------- perturbation operators
HOMO = {"a": "а", "e": "е", "o": "о", "p": "р", "c": "с", "x": "х", "y": "у", "i": "і", "s": "ѕ", "j": "ј", "A": "А",
        "E": "Е", "O": "О", "P": "Р", "C": "С", "T": "Т", "H": "Н", "K": "К", "M": "М", "B": "В"}
LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "g": "9"}
DIAC = {"a": "á", "e": "é", "i": "í", "o": "ó", "u": "ú", "n": "ñ", "c": "ç", "A": "Á", "E": "É", "I": "Í", "O": "Ó", "U": "Ú"}
ZW = ["​", "‌", "‍", "⁠", "﻿"]
UPSIDE = dict(zip("abcdefghijklmnopqrstuvwxyz", "ɐqɔpǝɟƃɥᴉɾʞlɯuodbɹsʇnʌʍxʎz"))


def p_homoglyph(t, r=0.5):
    return "".join(HOMO[c] if c in HOMO and rng.random() < r else c for c in t)


def p_leet(t, r=0.6):
    return "".join(LEET[c.lower()] if c.lower() in LEET and rng.random() < r else c for c in t)


def p_diacritics(t, r=0.7):
    return "".join(DIAC[c] if c in DIAC and rng.random() < r else c for c in t)


def p_zero_width(t, r=0.3):
    return "".join(c + (rng.choice(ZW) if rng.random() < r else "") for c in t)


def p_fullwidth(t):
    return "".join(chr(ord(c) + 0xFEE0) if "!" <= c <= "~" else ("　" if c == " " else c) for c in t)


def p_spaces(t):
    return " ".join(t) if len(t) < 400 else re.sub(r"(\w)", r"\1 ", t)


def p_upside(t):
    return "".join(UPSIDE.get(c.lower(), c) for c in reversed(t))


def p_tags(t):
    # Unicode tag characters (invisible) carrying the text, plus a visible benign-looking prefix
    return "".join(chr(0xE0000 + ord(c)) if " " <= c <= "~" else c for c in t)


def p_emoji(t):
    # variation selectors appended to an emoji
    return "😊" + "".join(chr(0xFE00 + (ord(c) % 16)) + c for c in t)


def p_underline(t):
    return "".join(c + "̲" if c.isalpha() else c for c in t)


def p_bidi(t):
    return "‮" + t[::-1] + "‬"


def p_typos(t, r=0.08):
    out = []
    for c in t:
        x = rng.random()
        if c.isalpha() and x < r / 3:
            continue  # deletion
        if c.isalpha() and x < 2 * r / 3:
            out.append(c + c)  # duplication
        elif c.isalpha() and x < r:
            out.append(rng.choice("abcdefghijklmnopqrstuvwxyz"))  # substitution
        else:
            out.append(c)
    return "".join(out)


OPS = [p_homoglyph, p_leet, p_diacritics, p_zero_width, p_fullwidth, p_spaces, p_upside, p_tags, p_emoji,
       p_underline, p_bidi, p_typos]

new = []
# ---------------------------------------------------------------- gen v2
stats = dict(calls=0, parsed=0, bad=0)
ev_rows = []
for f in GEN:
    for line in open(f):
        r = json.loads(line); stats["calls"] += 1
        p, a = r["parsed"], r["args"]
        if not isinstance(p, dict):
            continue
        stats["parsed"] += 1
        lang = LANG.get(a.get("lang"), "xx")
        held = rng.random() < 0.08
        out = []
        if r["task"] == "framing":
            out += [rec(t, 0, "gen2_framing/benign", "direct", lang) for t in p.get("benign") or [] if isinstance(t, str) and len(t) > 20]
            out += [rec(t, 1, "gen2_framing/jailbreak", "direct", lang) for t in p.get("jailbreak") or [] if isinstance(t, str) and len(t) > 20]
        else:
            legit, hij, span = p.get("legit"), p.get("hijacked"), p.get("injection_span")
            if isinstance(legit, str) and len(legit) > 50:
                out.append(rec(legit, 0, "gen2_hijack/legit", "indirect", lang))
            ok = isinstance(hij, str) and len(hij) > 50 and norm(hij) != norm(legit or "")
            if ok and isinstance(span, str) and len(span) > 10:
                ok = norm(span)[:60] in norm(hij)
            if ok:
                out.append(rec(hij, 1, "gen2_hijack/hijacked", "indirect", lang))
            else:
                stats["bad"] += 1
        (ev_rows if held else new).extend(out)
print(stats, "gen2 rows", len(new), "held", len(ev_rows), flush=True)

# ---------------------------------------------------------------- v1 + evasion augmentation
tr1 = pd.read_parquet(f"{V1}/train.parquet"); va1 = pd.read_parquet(f"{V1}/val.parquet")
if DROP:
    tr1 = tr1[~tr1.source.str.match(DROP)]; va1 = va1[~va1.source.str.match(DROP)]
short = tr1[(tr1.text.str.len() < 700) & (tr1.kind == "direct")]
atk = short[short.label == 1].sample(9000, random_state=3)
ben = short[short.label == 0].sample(9000, random_state=4)
for df, lab in [(atk, 1), (ben, 0)]:
    for t, lg in zip(df.text, df.lang):
        op = rng.choice(OPS)
        new.append(rec(op(t), lab, f"augment/{op.__name__[2:]}", "direct", lg))
# partial perturbation of injected docs: perturb only a window of the text
docs = tr1[(tr1.kind == "indirect") & (tr1.text.str.len() < 3000)]
for df, lab in [(docs[docs.label == 1].sample(3000, random_state=5), 1), (docs[docs.label == 0].sample(3000, random_state=6), 0)]:
    for t, lg in zip(df.text, df.lang):
        op = rng.choice([p_homoglyph, p_zero_width, p_leet, p_diacritics, p_typos, p_fullwidth])
        s = rng.randint(0, max(0, len(t) - 400)); e = s + rng.randint(100, 400)
        new.append(rec(t[:s] + op(t[s:e]) + t[e:], lab, f"augment_doc/{op.__name__[2:]}", "indirect", lg))

# ---------------------------------------------------------------- merge, decontaminate
evals = {os.path.basename(p)[:-8]: pd.read_parquet(p) for p in glob.glob(f"{V1}/eval/*.parquet")}
evals["synthetic_v2_test"] = pd.DataFrame(ev_rows).drop_duplicates("text")
eval_keys = set()
for df in evals.values():
    eval_keys |= set(df.text.map(norm))
nd = pd.DataFrame(new); nd = nd[nd.text.str.len() > 3]; nd["k"] = nd.text.map(norm)
old = set(tr1.text.map(norm)) | set(va1.text.map(norm))
before = len(nd)
nd = nd[~nd.k.isin(eval_keys) & ~nd.k.isin(old)]
conf = nd.groupby("k").label.nunique()
nd = nd[~nd.k.isin(set(conf[conf > 1].index))].drop_duplicates("k").sample(frac=1.0, random_state=7)
print("new kept", len(nd), "of", before, flush=True)
nval = int(0.03 * len(nd))
cols = ["text", "label", "source", "kind", "lang"]
tr = pd.concat([tr1[cols], nd.iloc[nval:][cols]]).sample(frac=1.0, random_state=8).reset_index(drop=True)
va = pd.concat([va1[cols], nd.iloc[:nval][cols]]).reset_index(drop=True)
tr.to_parquet(f"{OUT}/train.parquet"); va.to_parquet(f"{OUT}/val.parquet")
for n, df in evals.items():
    df[cols].to_parquet(f"{OUT}/eval/{n}.parquet")
summary = dict(train=len(tr), val=len(va), train_pos=int(tr.label.sum()),
               by_source=tr.groupby("source").label.agg(["count", "sum"]).reset_index().values.tolist(),
               evals={k: [len(v), int(v.label.sum())] for k, v in evals.items()})
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1, default=str)
print(json.dumps(summary, default=str)[:2500])
