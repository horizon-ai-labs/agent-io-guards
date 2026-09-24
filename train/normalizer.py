"""Obfuscation-resistant tokenizer normalizer, prepended to the model's own normalizer.

1. Decode "smuggled" text so the classifier sees what the downstream LLM can read:
   Unicode tag characters U+E0020..E007E -> ASCII; variation selectors used as bytes
   (U+FE00..FE0F = 0..15, U+E0100..E01EF = 16..255) -> the printable ASCII they encode.
2. NFKC (folds full-width and compatibility forms).
3. Strip remaining invisible / formatting characters: zero-width, bidi controls, soft hyphen,
   leftover tag/variation selectors, combining overlay and underline marks.
Homoglyphs are NOT mapped (Cyrillic/Greek are legitimate scripts); the model handles them.
"""
from tokenizers import normalizers, Regex

INVISIBLE = ("[­᠎​-‏‪-‮⁠-⁤⁦-⁩﻿︀-️"
             "\U000e0000-\U000e007f\U000e0100-\U000e01ef̱̅-̸⃐-⃿]")


def decode_rules():
    rules = []
    for c in range(0x20, 0x7F):
        rules.append(normalizers.Replace(chr(0xE0000 + c), chr(c)))          # tag characters
        if c >= 16:
            rules.append(normalizers.Replace(chr(0xE0100 + c - 16), chr(c)))  # variation selector supplement as byte
    return rules


def add_normalizer(tok):
    """tok: a transformers fast tokenizer; modifies it in place and returns it."""
    bk = tok.backend_tokenizer
    old = bk.normalizer
    seq = decode_rules() + [normalizers.NFKC(), normalizers.Replace(Regex(INVISIBLE), "")]
    if old is not None:
        seq.append(old)
    bk.normalizer = normalizers.Sequence(seq)
    return tok
