"""Minimal helpers for Horizon-Labs PII redactor models.

    from redact import PIIRedactor
    r = PIIRedactor("Horizon-Labs/pii-redactor-small")
    r.redact("Mail anna.muller@example.org or call +49 30 1234567")
    # -> 'Mail [EMAIL] or call [PHONE]'
"""
from transformers import pipeline


class PIIRedactor:
    def __init__(self, model="Horizon-Labs/pii-redactor-small", device=None, threshold=0.5, labels=None):
        self.nlp = pipeline("token-classification", model=model, aggregation_strategy="simple", device=device, stride=128)
        self.threshold = threshold
        self.labels = set(labels) if labels else None

    def entities(self, text):
        raw = sorted(self.nlp(text), key=lambda e: e["start"])
        if self.labels is not None:
            raw = [e for e in raw if e["entity_group"] in self.labels]
        # merge same-label pieces separated only by short spacing / punctuation (e.g. "+49 30 1234567")
        out = []
        for e in raw:
            s, t = int(e["start"]), int(e["end"])
            if out and out[-1]["entity_group"] == e["entity_group"] and s - out[-1]["end"] <= 3 \
                    and not any(c.isalpha() for c in text[out[-1]["end"]:s]):
                out[-1]["end"] = t; out[-1]["score"] = max(out[-1]["score"], float(e["score"]))
            else:
                out.append(dict(entity_group=e["entity_group"], start=s, end=t, score=float(e["score"])))
        res = []
        for e in out:
            # the tokenizer attaches the preceding space to a word; keep whitespace out of the span
            while e["start"] < e["end"] and text[e["start"]].isspace(): e["start"] += 1
            while e["end"] > e["start"] and text[e["end"] - 1].isspace(): e["end"] -= 1
            if e["score"] >= self.threshold and e["end"] > e["start"]:
                e["text"] = text[e["start"]:e["end"]]
                res.append(e)
        return res

    def redact(self, text, placeholder="[{label}]"):
        ents = self.entities(text)
        for e in reversed(ents):
            text = text[:e["start"]] + placeholder.format(label=e["entity_group"]) + text[e["end"]:]
        return text


def presidio_recognizer(model="Horizon-Labs/pii-redactor-small", **kw):
    """Presidio EntityRecognizer backed by the model. Entity names follow Presidio where one exists."""
    from presidio_analyzer import EntityRecognizer, RecognizerResult
    TO_PRESIDIO = {"PERSON": "PERSON", "EMAIL": "EMAIL_ADDRESS", "PHONE": "PHONE_NUMBER", "LOCATION": "LOCATION",
                   "STREET_ADDRESS": "LOCATION", "DATE": "DATE_TIME", "DATE_OF_BIRTH": "DATE_TIME", "CREDIT_CARD": "CREDIT_CARD",
                   "BANK_ACCOUNT": "IBAN_CODE", "IP_ADDRESS": "IP_ADDRESS", "URL": "URL", "NATIONAL_ID": "US_SSN",
                   "PASSPORT": "US_PASSPORT", "DRIVER_LICENSE": "US_DRIVER_LICENSE", "MEDICAL_ID": "MEDICAL_LICENSE"}
    red = PIIRedactor(model, **kw)

    class HorizonPIIRecognizer(EntityRecognizer):
        def load(self):
            pass

        def analyze(self, text, entities, nlp_artifacts=None):
            res = []
            for e in red.entities(text):
                name = TO_PRESIDIO.get(e["entity_group"], e["entity_group"])
                if not entities or name in entities:
                    res.append(RecognizerResult(name, e["start"], e["end"], e["score"]))
            return res

    labels = sorted(set(TO_PRESIDIO.get(l.split("-", 1)[-1], l.split("-", 1)[-1]) for l in red.nlp.model.config.id2label.values() if l != "O"))
    return HorizonPIIRecognizer(supported_entities=labels, name="HorizonPIIRecognizer")
