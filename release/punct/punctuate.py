"""Restore punctuation with Horizon-Labs/punctuation-restoration-* in any language (incl. Chinese/Japanese, no word splitting).
    from punctuate import Punctuator
    p = Punctuator("Horizon-Labs/punctuation-restoration-small")
    p("hello how are you i am fine thanks")   # -> "hello, how are you? i am fine, thanks."
Existing marks . , ; : ! ? are removed first (as in deepmultilingualpunctuation), except inside numbers (3.14, 1,000, 10:30).
Long texts are processed in overlapping chunks.
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


class Punctuator:
    def __init__(self, model="Horizon-Labs/punctuation-restoration-small", device=None, chunk_tokens=400, overlap_tokens=50):
        self.tok = AutoTokenizer.from_pretrained(model)
        self.model = AutoModelForTokenClassification.from_pretrained(model).eval()
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.id2label = self.model.config.id2label
        self.chunk, self.overlap = chunk_tokens, overlap_tokens

    @torch.no_grad()
    def labels(self, text):
        """-> {char_end: label} for the token ends where a mark should follow."""
        enc = self.tok(text, return_offsets_mapping=True, add_special_tokens=False)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        out, start = {}, 0
        while start < len(ids):
            end = min(len(ids), start + self.chunk)
            x = torch.tensor([[self.tok.cls_token_id] + ids[start:end] + [self.tok.sep_token_id]], device=self.device)
            pred = self.model(input_ids=x).logits[0, 1:-1].argmax(-1).tolist()
            keep_to = end if end == len(ids) else end - self.overlap
            for k in range(start, keep_to):
                lab = self.id2label[pred[k - start]]
                s, e = offs[k]
                if e > s:
                    out[e] = lab
            if end == len(ids):
                break
            start = keep_to
        return out

    def __call__(self, text):
        text = re.sub(r"\s+", " ", strip_marks(text)).strip()
        if not text:
            return text
        labs = self.labels(text)
        res, prev = [], 0
        for e in sorted(labs):
            lab = labs[e]
            # a mark goes only at a word end (next char is a space or the end), or after any character in CJK text
            if lab == "0" or not (e == len(text) or text[e] == " " or CJK.match(text[e - 1])):
                continue
            mark = mark_for(lab, text[e - 1])
            res.append(text[prev:e] + mark); prev = e
        return "".join(res) + text[prev:]
