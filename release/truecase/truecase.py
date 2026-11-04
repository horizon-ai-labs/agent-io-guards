"""Restore punctuation AND casing with Horizon-Labs/multilingual-punctuation-truecase-* (lowercase ASR-style text in, any language).
    from truecase import Truecaser
    t = Truecaser("Horizon-Labs/multilingual-punctuation-truecase-small")
    t("my name is anna i live in berlin and work for the united nations in new york")
    # -> "My name is Anna. I live in Berlin and work for the United Nations in New York."
The input is lowercased first (the model is trained on lowercased text; your casing is not used). Existing marks . , ; : ! ? are
removed first, except inside numbers (3.14, 1,000, 10:30). Labels are '<mark>|<case>': mark in 0 . , ? - : (as
punctuation-restoration-*), case lower / Cap (first letter upper) / UPPER (all upper). Long texts are processed in overlapping chunks.
"""
import re
import torch
from transformers import AutoModelForTokenClassification, AutoTokenizer

FULLWIDTH = {".": "。", ",": "，", "?": "？", ":": "：", "-": "——"}
CJK = re.compile(r"[぀-ヿ㐀-鿿豈-﫿]")
# script-specific marks, chosen by the character before the mark
SCRIPT = [(re.compile(r"[ऀ-৿]"), {".": "।"}),                    # Devanagari, Bengali
          (re.compile(r"[؀-ۿݐ-ݿ]"), {",": "،", "?": "؟"}),  # Arabic script
          (re.compile(r"[Ͱ-Ͽἀ-῿]"), {"?": ";"}),        # Greek
          (re.compile(r"[ሀ-፿]"), {".": "።", ",": "፣"})]            # Ethiopic
MARKS = re.compile(r"[.,;:!?。，、；：！？؟،¿¡…।॥۔።፣፧]")


def mark_for(lab, prev_char):
    if CJK.match(prev_char):
        return FULLWIDTH[lab]
    for rx, m in SCRIPT:
        if rx.match(prev_char):
            return m.get(lab, " -" if lab == "-" else lab)
    return " -" if lab == "-" else lab


def strip_marks(t):
    return MARKS.sub(lambda m: m.group() if m.group() in ".,:" and 0 < m.start() < len(t) - 1 and t[m.start() - 1].isdigit()
                     and t[m.start() + 1].isdigit() else "", t)


CASES = ["lower", "Cap", "UPPER"]


def lower_keep_len(t):
    return "".join(c.lower() if len(c.lower()) == 1 else c for c in t)


def apply_case(word, case):
    if case == "UPPER":
        return word.upper() if len(word.upper()) == len(word) else word
    if case == "Cap":
        for i, ch in enumerate(word):
            if ch.lower() != ch.upper():
                return word[:i] + ch.upper() + word[i + 1:] if len(ch.upper()) == 1 else word
    return word


class Truecaser:
    def __init__(self, model="Horizon-Labs/multilingual-punctuation-truecase-small", device=None, chunk_tokens=400, overlap_tokens=50):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label
        self.chunk, self.overlap = chunk_tokens, overlap_tokens

    @torch.no_grad()
    def labels(self, text):
        """-> {char_end: '<mark>|<case>'} for every token end."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        out, start = {}, 0
        while start < len(ids):
            end = min(len(ids), start + self.chunk)
            x = torch.tensor([[self.tok.cls_token_id] + ids[start:end] + [self.tok.sep_token_id]], device=self.device)
            pred = self.model(input_ids=x).logits[0, 1:-1].argmax(-1).tolist()
            keep_to = end if end == len(ids) else end - self.overlap
            for k in range(start, keep_to):
                s, e = offs[k]
                if e > s:
                    out[e] = self.id2label[pred[k - start]]
            if end == len(ids):
                break
            start = keep_to
        return out

    def __call__(self, text):
        text = lower_keep_len(re.sub(r"\s+", " ", strip_marks(text)).strip())
        if not text:
            return text
        labs = self.labels(text)
        res, prev = [], 0
        for e in sorted(labs):
            # a label counts only at a word end (next char is a space or the end), or after any character in CJK text
            if not (e == len(text) or text[e] == " " or CJK.match(text[e - 1])):
                continue
            mark, case = labs[e].split("|") if "|" in labs[e] else (labs[e], "lower")
            s = max(prev, text.rfind(" ", 0, e) + 1)
            res.append(text[prev:s] + apply_case(text[s:e], case) + ("" if mark == "0" else mark_for(mark, text[e - 1]))); prev = e
        return "".join(res) + text[prev:]
