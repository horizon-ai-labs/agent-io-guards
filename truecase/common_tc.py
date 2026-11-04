"""Punctuation + truecasing: joint per-word labels = the punctuation mark that follows the word (punct/common.py: 0 . , ? - :) x the
word's casing. Casing classes from the original text: 'lower' (no upper-case first letter), 'Cap' (first cased letter upper, not all
upper), 'UPPER' (all cased letters upper, >= 2 of them). Words without cased letters (CJK, Arabic, ...) are 'lower'.
Inputs are always lowercased (ASR-style); the model restores marks and casing together."""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "punct"))
from common import LABELS as PUNCT, NOSPACE, lower_keep_len  # noqa: F401
CASES = ["lower", "Cap", "UPPER"]
LABELS = [f"{p}|{c}" for p in PUNCT for c in CASES]
L2I = {l: i for i, l in enumerate(LABELS)}


def case_of(word):
    cased = [ch for ch in word if ch.lower() != ch.upper()]
    if not cased:
        return 0
    if len(cased) >= 2 and all(ch.isupper() for ch in cased):
        return 2
    return 1 if cased[0].isupper() else 0


def unit_cases(text, ends, nospace):
    """case class per unit of a cased text; units end at `ends` (exclusive offsets)."""
    out, prev = [], 0
    for e in ends:
        if nospace:
            s = e - 1
        else:
            s = text.rfind(" ", 0, e) + 1
        out.append(case_of(text[max(s, prev):e])); prev = e
    return out


def apply_case(word, c):
    if c == 2:
        return word.upper()
    if c == 1:
        for i, ch in enumerate(word):
            if ch.lower() != ch.upper():
                return word[:i] + ch.upper() + word[i + 1:]
    return word
