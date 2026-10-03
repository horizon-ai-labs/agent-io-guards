"""Punctuation restoration: shared text normalisation (training data, evaluation, inference helper).
Labels = those of oliverguhr/fullstop-punctuation-multilang-large and kredor/punctuate-all, so the models are drop-ins
for the `deepmultilingualpunctuation` package: "0" (none) . , ? - :   - each is the mark that FOLLOWS a word.
Mapping: ! ; … and script full stops (。！．।॥۔።။։) -> "."; ，、،፣ -> ","; ？؟፧ (and ";" in Greek) -> "?"; ：-> ":";
a free-standing dash (- – — --) -> "-". Inverted marks ¿ ¡ are removed. Marks between digits (3.14, 1,000, 10:30) are text.
Units: whitespace words; for Chinese and Japanese every non-space character is a unit.
"""
import re

LABELS = ["0", ".", ",", "?", "-", ":"]
L2I = {l: i for i, l in enumerate(LABELS)}
NOSPACE = {"cmn_Hani", "jpn_Jpan", "zho_Hans", "zho_Hant", "yue_Hani", "wuu_Hani"}
SKIP = {"tha_Thai", "lao_Laoo", "khm_Khmr", "mya_Mymr", "bod_Tibt"}   # no comparable punctuation conventions
MAP = {".": ".", "!": ".", ";": ".", "…": ".", "。": ".", "！": ".", "．": ".", "।": ".", "॥": ".", "۔": ".", "።": ".", "။": ".",
       "։": ".", "；": ".", "؛": ".", "｡": ".",
       ",": ",", "，": ",", "、": ",", "،": ",", "፣": ",", "､": ",",
       "?": "?", "？": "?", "؟": "?", "፧": "?", ";": ".",
       ":": ":", "：": ":"}
DROP = set("¿¡")
DASH = re.compile(r"^(?:-{1,2}|–|—)$")
PRI = {"?": 4, ".": 3, ":": 2, "-": 1, ",": 1, "0": 0}


def _marks(lang):
    m = dict(MAP)
    if lang == "ell_Grek":
        m[";"] = "?"; m[";"] = "?"
    return m


def _between_digits(s, i):
    return 0 < i < len(s) - 1 and s[i - 1].isdigit() and s[i + 1].isdigit() and s[i] in ".,:"


def units(text, lang):
    """original punctuated text -> (clean text, unit ends (char offsets in clean text), labels (strings)).
    The clean text is what a user would feed: target marks removed, other symbols (quotes, brackets, hyphens) kept."""
    m = _marks(lang)
    if lang in NOSPACE:
        out, ends, labs = [], [], []
        for i, c in enumerate(text):
            if c in DROP:
                continue
            if c in m and not _between_digits(text, i) or (c in "-–—" and lang in NOSPACE and c != "-"):
                lab = m.get(c, "-")
                if labs and PRI[lab] > PRI[labs[-1]]:
                    labs[-1] = lab
                continue
            if c.isspace():
                if out and not out[-1].isspace():
                    out.append(" ")
                continue
            out.append(c)
            ends.append(len(out)); labs.append("0")
        return "".join(out).strip(), ends, labs
    words, labs, dots = [], [], []
    for w in text.split():
        if DASH.match(w):
            if words and labs[-1] == "0":
                labs[-1] = "-"
            continue
        core, lab, trailing = [], "0", True
        for i in range(len(w) - 1, -1, -1):   # label = strongest mark in the trailing non-alphanumeric run
            c = w[i]
            if c.isalnum():
                trailing = False
            if c in m and not _between_digits(w, i):
                if trailing and PRI[m[c]] > PRI[lab]:
                    lab = m[c]
                continue
            if c in DROP:
                continue
            core.append(c)
        core = "".join(reversed(core))
        if not core.strip() or not any(ch.isalnum() for ch in core) and not core.strip("\"'«»„“”‘’()[]{}*"):
            if words and PRI[lab] > PRI[labs[-1]]:
                labs[-1] = lab
            if not core.strip():
                continue
        words.append(core); labs.append(lab); dots.append(lab == "." and w.rstrip("\"'»”’)]").endswith("."))
    for i in range(len(words) - 1):   # "." before a lowercase word is an abbreviation (e.g., U.S. policy), not a full stop
        if dots[i] and words[i + 1][:1].islower():
            labs[i] = "0"
    clean, ends, pos = [], [], 0
    for w in words:
        if clean:
            pos += 1
        pos += len(w); ends.append(pos); clean.append(w)
    return " ".join(clean), ends, labs


def lower_keep_len(t):
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in t)


def token_labels(offsets, ends, labs):
    """token offsets (fast tokenizer) -> label id per token: a token that ends where a unit ends gets the unit's label,
    other tokens -100 (inside a unit, special tokens)."""
    at = dict(zip(ends, labs))
    out = []
    for s, e in offsets:
        out.append(L2I[at[e]] if e > s and e in at else -100)
    return out


def unit_preds(offsets, pred_ids, ends):
    """per-token predicted label ids -> label string per unit (the token ending at the unit end; "0" if none does)."""
    at = {e: p for (s, e), p in zip(offsets, pred_ids) if e > s}
    return [LABELS[at[e]] if e in at else "0" for e in ends]


def restore(clean, ends, labs):
    out, prev = [], 0
    for e, l in zip(ends, labs):
        out.append(clean[prev:e] + ("" if l == "0" else (" -" if l == "-" else l))); prev = e
    return "".join(out) + clean[prev:]
