"""Build PII / secrets span data (v0) from permissively licensed sources, unified taxonomy.

Output parquet (text, spans_json, source, lang, split): $OUT/train.parquet, $OUT/val.parquet, $OUT/eval/<name>.parquet
spans_json = JSON list of [start, end, LABEL] character spans.

Sources
  ai4privacy/pii-masking-openpii-1.5m  CC-BY-4.0 (card), 30 languages, 19 labels   (sampled, language-balanced)
  nvidia/Nemotron-PII                  CC-BY-4.0, en (us + intl locales), ~55 labels
  gretelai/gretel-pii-masking-en-v1    Apache-2.0, en
"""
import ast, json, os, random, re
from collections import Counter
import pandas as pd
from datasets import load_dataset

OUT = os.environ.get("OUT", "pii/v0")
N_OPENPII = int(os.environ.get("N_OPENPII", 240000))
os.makedirs(f"{OUT}/eval", exist_ok=True)
rng = random.Random(0)

# Unified, Presidio-aligned taxonomy. None = not tagged (treated as O).
MAP = {
    # people / orgs
    "GIVENNAME": "PERSON", "SURNAME": "PERSON", "first_name": "PERSON", "last_name": "PERSON", "name": "PERSON",
    "TITLE": None, "company_name": "ORGANIZATION",
    # contact
    "EMAIL": "EMAIL", "email": "EMAIL", "TELEPHONENUM": "PHONE", "phone_number": "PHONE", "fax_number": "PHONE",
    "url": "URL", "user_name": "USERNAME",
    # location
    "STREET": "STREET_ADDRESS", "BUILDINGNUM": "STREET_ADDRESS", "street_address": "STREET_ADDRESS", "address": "STREET_ADDRESS",
    "CITY": "LOCATION", "city": "LOCATION", "state": "LOCATION", "county": "LOCATION", "country": "LOCATION",
    "ZIPCODE": "POSTCODE", "postcode": "POSTCODE", "coordinate": "COORDINATE",
    # time / demographics
    "DATE": "DATE", "date": "DATE", "date_time": "DATE", "TIME": "DATE", "time": "DATE",
    "date_of_birth": "DATE_OF_BIRTH", "AGE": "AGE", "age": "AGE",
    # government ids
    "SOCIALNUM": "NATIONAL_ID", "ssn": "NATIONAL_ID", "IDCARDNUM": "NATIONAL_ID", "national_id": "NATIONAL_ID",
    "TAXNUM": "TAX_ID", "tax_id": "TAX_ID", "PASSPORTNUM": "PASSPORT", "DRIVERLICENSENUM": "DRIVER_LICENSE",
    "certificate_license_number": "LICENSE_NUMBER",
    # financial
    "CREDITCARDNUMBER": "CREDIT_CARD", "credit_debit_card": "CREDIT_CARD", "credit_card_number": "CREDIT_CARD",
    "cvv": "CREDIT_CARD_CVV", "account_number": "BANK_ACCOUNT", "bank_routing_number": "BANK_ACCOUNT",
    "swift_bic": "BANK_ACCOUNT", "iban": "BANK_ACCOUNT",
    # technical / secrets
    "ipv4": "IP_ADDRESS", "ipv6": "IP_ADDRESS", "mac_address": "MAC_ADDRESS", "api_key": "SECRET",
    "password": "PASSWORD", "pin": "PASSWORD", "http_cookie": "SECRET", "device_identifier": "DEVICE_ID",
    # other identifiers
    "customer_id": "ACCOUNT_ID", "employee_id": "ACCOUNT_ID", "unique_id": "ACCOUNT_ID", "unique_identifier": "ACCOUNT_ID",
    "medical_record_number": "MEDICAL_ID", "health_plan_beneficiary_number": "MEDICAL_ID",
    "license_plate": "VEHICLE_ID", "vehicle_identifier": "VEHICLE_ID", "biometric_identifier": None,
    # sensitive attributes / low-value categories: not tagged in v0
    "SEX": None, "gender": None, "occupation": None, "education_level": None, "employment_status": None,
    "language": None, "religious_belief": None, "political_view": None, "sexuality": None, "race_ethnicity": None,
    "blood_type": None,
    # extra openpii labels seen in the data (not in its card's list of 19)
    "GENDER": None, "URL": "URL", "USERNAME": "USERNAME", "IPV4": "IP_ADDRESS", "IPV6": "IP_ADDRESS", "PASSWORD": "PASSWORD",
    "ACCOUNTNUM": "BANK_ACCOUNT", "ORGANISATION": "ORGANIZATION", "COUNTRY": "LOCATION", "AMOUNT": None, "CURRENCY": None,
    "BANKNAME": "ORGANIZATION", "JOBTITLE": None, "HOSPITALNAME": "ORGANIZATION",
}
unmapped = Counter()


def spans_from(items, text):
    out = []
    for s, e, lab in items:
        if lab not in MAP:
            unmapped[lab] += 1
            continue
        L = MAP[lab]
        if L is None or not (0 <= s < e <= len(text)):
            continue
        # trim whitespace inside span
        while s < e and text[s].isspace(): s += 1
        while e > s and text[e - 1].isspace(): e -= 1
        if e > s:
            out.append([s, e, L])
    out.sort()
    # merge adjacent same-label spans separated by a single space (e.g. GIVENNAME SURNAME -> PERSON)
    merged = []
    for sp in out:
        if merged and merged[-1][2] == sp[2] and 0 <= sp[0] - merged[-1][1] <= 1 and text[merged[-1][1]:sp[0]].strip() == "":
            merged[-1][1] = sp[1]
        elif merged and sp[0] < merged[-1][1]:
            continue  # overlap: keep first
        else:
            merged.append(sp)
    return merged


def rec(text, spans, source, lang, split):
    return dict(text=text, spans_json=json.dumps(spans), source=source, lang=lang, split=split)


rows = []
# ---- openpii: language-balanced sample
op = load_dataset("ai4privacy/pii-masking-openpii-1.5m", split="train")
by_lang = {}
for i, l in enumerate(op["language"]):
    by_lang.setdefault(l, []).append(i)
per = N_OPENPII // len(by_lang)
idx = []
for l, ii in by_lang.items():
    rng.shuffle(ii)
    idx += ii[: per * (3 if l == "en" else 1)]
for r in op.select(idx):
    t = r["source_text"]
    rows.append(rec(t, spans_from([(m["start"], m["end"], m["label"]) for m in r["privacy_mask"]], t), "openpii", r["language"], "train"))
opv = load_dataset("ai4privacy/pii-masking-openpii-1.5m", split="validation").shuffle(seed=0).select(range(6000))
for r in opv:
    t = r["source_text"]
    rows.append(rec(t, spans_from([(m["start"], m["end"], m["label"]) for m in r["privacy_mask"]], t), "openpii", r["language"], "eval_openpii"))

# ---- Nemotron-PII
for split, tag in [("train", "train"), ("test", "eval_nemotron")]:
    for r in load_dataset("nvidia/Nemotron-PII", split=split):
        t = r["text"]; sp = r["spans"]
        sp = ast.literal_eval(sp) if isinstance(sp, str) else sp
        rows.append(rec(t, spans_from([(m["start"], m["end"], m["label"]) for m in sp], t), "nemotron", "en", tag))

# ---- Gretel: entities are strings + types, locate them in text
for split, tag in [("train", "train"), ("test", "eval_gretel")]:
    for r in load_dataset("gretelai/gretel-pii-masking-en-v1", split=split):
        t = r["text"]; items = []
        for m in ast.literal_eval(r["entities"]):
            ent = m["entity"].strip()
            if len(ent) < 2:
                continue
            for mm in re.finditer(re.escape(ent), t):
                items.append((mm.start(), mm.end(), m["types"][0]))
        rows.append(rec(t, spans_from(items, t), "gretel", "en", tag))

df = pd.DataFrame(rows)
df = df[df.text.str.len().between(20, 20000)]
df = df.drop_duplicates("text")
print("unmapped labels:", unmapped.most_common(20))
tr = df[df.split == "train"].sample(frac=1.0, random_state=1)
nval = 4000
tr.iloc[nval:].to_parquet(f"{OUT}/train.parquet"); tr.iloc[:nval].to_parquet(f"{OUT}/val.parquet")
for tag in df.split.unique():
    if tag.startswith("eval_"):
        df[df.split == tag].to_parquet(f"{OUT}/eval/{tag[5:]}.parquet")
lab = Counter(l for s in tr.spans_json for _, _, l in json.loads(s))
summary = dict(train=len(tr) - nval, by_source=tr.source.value_counts().to_dict(), labels=lab.most_common(),
               langs=tr.lang.value_counts().to_dict(), evals=df[df.split != "train"].split.value_counts().to_dict())
json.dump(summary, open(f"{OUT}/summary.json", "w"), indent=1)
print(json.dumps(summary)[:3000])
