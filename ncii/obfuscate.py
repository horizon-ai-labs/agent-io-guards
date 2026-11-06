"""Character-level obfuscation families for the NCII guard: training augmentation (applied to BOTH classes, so obfuscation itself is not
a signal) and a per-family robustness test. Several families are undone by the tokenizer normalizer (train/normalizer.py: NFKC,
invisible characters), the rest must be learned. Homoglyphs map Latin letters to Cyrillic/Greek lookalikes (applied only to Latin-script
text; the normalizer does not fold them, because Cyrillic and Greek are real scripts)."""
import random

LEET = {"a": "4", "e": "3", "i": "1", "o": "0", "s": "5", "t": "7", "b": "8", "g": "9"}
LEET_HEAVY = {**LEET, "a