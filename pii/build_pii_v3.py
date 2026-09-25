"""PII data v3 = v0 + programmatic payment / bank snippets (no LLM, exact labels).

Why: v1.0 misses card expiry dates without a "date" keyword, CVVs next to "security code" / "cvc" / "cryptogramme",
and sometimes the first digit of unspaced card numbers; IBANs come out as mixed types. v1/v2 (Qwen documents) hurt
external recall, so v3 keeps v0 unchanged and adds a small, targeted set of short snippets.
Labels: card number CREDIT_CARD, CVV/CVC CREDIT_CARD_CVV, expiry DATE (the taxonomy has no expiry type), IBAN / BIC /
account numbers BANK_ACCOUNT, names PERSON, emails EMAIL. Amounts, prices, times and quantities stay O.
Card numbers are Luhn-valid test-style numbers and IBANs have valid check digits; none are real accounts.
Train snippets use TRAIN templates; eval/synthetic_payment.parquet uses held-out EVAL templates and keywords.
env: V0=<pii_v0 dir> OUT=<dir> N=<train snippets>
"""
import json, os, random
import pandas as pd

V0, OUT, N = os.environ["V0"], os.environ["OUT"], int(os.environ.get("N", 6000))
rng = random.Random(11)


def luhn_complete(prefix, length):
    d = [int(c) for c in prefix] + [rng.randint(0, 9) for _ in range(length - len(prefix) - 1)]
    s = 0
    for i, x in enumerate(reversed(d)):
        if i % 2 == 0:
            x *= 2
            x = x - 9 if x > 9 else x
        s += x
    return "".join(map(str, d)) + str((10 - s % 10) % 10)


def card():
    brand = rng.choice(["visa", "visa", "mc", "mc", "amex", "discover", "jcb"])
    if brand == "amex":
        n = luhn_complete(rng.choice(["34", "37"]), 15); groups = [n[:4], n[4:10], n[10:]]
    else:
        pre = {"visa": "4", "mc": str(rng.randint(51, 55)), "discover": "6011", "jcb": "35" + str(rng.randint(28, 89))}[brand]
        n = luhn_complete(pre, 16); groups = [n[i:i + 4] for i in range(0, 16, 4)]
    sep = rng.choice([" ", " ", "-", "", "", " "])
    return sep.join(groups) if sep else "".join(groups)


def expiry():
    m, y = rng.randint(1, 12), rng.randint(25, 34)
    return rng.choice([f"{m:02d}/{y}", f"{m:02d}/20{y}", f"{m:02d}-{y}", f"{m}/{y}", f"{m:02d}/{y}", f"{m:02d} / {y}"])


def cvv(amex=False):
    return "".join(str(rng.randint(0, 9)) for _ in range(4 if amex or rng.random() < 0.1 else 3))


IBAN_SPEC = {"DE": 18, "ES": 20, "FR": 23, "IT": 23, "NL": 14, "GB": 18, "PL": 24, "BE": 12, "AT": 16, "CH": 17, "PT": 21, "SE": 20}


def iban():
    c = rng.choice(list(IBAN_SPEC))
    L = IBAN_SPEC[c]
    if c in ("NL", "GB"):
        bban = "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + "".join(str(rng.randint(0, 9)) for _ in range(L - 4))
    else:
        bban = "".join(str(rng.randint(0, 9)) for _ in range(L))
    num = "".join(str(int(ch, 36)) for ch in bban + c + "00")
    s = c + f"{98 - int(num) % 97:02d}" + bban
    return " ".join(s[i:i + 4] for i in range(0, len(s), 4)) if rng.random() < 0.6 else s


def bic():
    return "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ") for _ in range(4)) + rng.choice(["DE", "FR", "ES", "IT", "GB", "NL"]) + \
        "".join(rng.choice("ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789") for _ in range(2)) + rng.choice(["", "XXX", "123"])


NAMES = ["Anna Schmidt", "Luca Rossi", "Marie Dubois", "Carlos García", "Sofia Nowak", "James O'Brien", "Yuki Tanaka",
         "Min-jun Kim", "Wei Zhang", "Olga Ivanova", "Fatima Zahra", "Priya Sharma", "Tomás Silva", "Emma de Vries",
         "Noah Müller", "Chloé Martin", "Mateo López", "Aylin Yılmaz", "Lars Nilsson", "Grace Okafor"]
AMOUNT = lambda: rng.choice([f"€{rng.randint(5, 900)}.{rng.randint(0, 99):02d}", f"${rng.randint(5, 2000)}.{rng.randint(0, 99):02d}",
                             f"{rng.randint(5, 900)},{rng.randint(0, 99):02d} €", f"£{rng.randint(5, 500)}", f"{rng.randint(1000, 90000)} ¥"])
TIME = lambda: f"{rng.randint(0, 23):02d}:{rng.choice(['00', '15', '30', '45'])}"

# (card keyword, expiry keyword, cvv keyword) per language; TRAIN and EVAL use disjoint keyword variants where possible
KW = {
    "en": dict(train=[("Card number", "Exp", "CVV"), ("card", "expires", "security code"), ("Visa", "valid thru", "CVC"),
                      ("cc", "exp date", "cvv2"), ("card no.", "expiry", "3-digit code")],
               eval=[("Credit card", "good thru", "card verification code"), ("Mastercard", "exp.", "CSC")]),
    "de": dict(train=[("Kartennummer", "gültig bis", "Prüfnummer"), ("Kreditkarte", "Ablaufdatum", "CVC")],
               eval=[("Karte", "Verfall", "Sicherheitscode")]),
    "fr": dict(train=[("Numéro de carte", "expire fin", "cryptogramme"), ("carte bancaire", "date d'expiration", "CVV")],
               eval=[("CB", "valable jusqu'au", "code de sécurité")]),
    "es": dict(train=[("Número de tarjeta", "caduca", "CVV"), ("tarjeta", "vencimiento", "código de seguridad")],
               eval=[("Tarjeta de crédito", "válida hasta", "CVC")]),
    "it": dict(train=[("Numero carta", "scadenza", "CVV")], eval=[("carta di credito", "scade", "codice di sicurezza")]),
    "pt": dict(train=[("Número do cartão", "validade", "CVV")], eval=[("cartão", "vence em", "código de segurança")]),
    "nl": dict(train=[("Kaartnummer", "geldig tot", "CVC")], eval=[("creditcard", "vervaldatum", "beveiligingscode")]),
    "pl": dict(train=[("Numer karty", "ważna do", "CVV")], eval=[("karta", "data ważności", "kod CVC")]),
    "ru": dict(train=[("Номер карты", "срок действия", "CVV")], eval=[("карта", "действует до", "код безопасности")]),
    "ja": dict(train=[("カード番号", "有効期限", "セキュリティコード")], eval=[("クレジットカード", "期限", "CVV")]),
    "ko": dict(train=[("카드번호", "유효기간", "CVC")], eval=[("신용카드", "만료일", "보안코드")]),
    "zh": dict(train=[("卡号", "有效期", "安全码")], eval=[("信用卡", "到期", "CVV")]),
}
IBAN_KW = {"en": ["IBAN", "account (IBAN)", "bank account"], "de": ["IBAN", "Konto"], "fr": ["IBAN", "RIB"], "es": ["IBAN", "cuenta"],
           "it": ["IBAN", "conto"], "pt": ["IBAN", "NIB"], "nl": ["IBAN", "rekeningnummer"], "pl": ["IBAN", "numer konta"],
           "ru": ["IBAN", "счёт"], "ja": ["IBAN", "口座"], "ko": ["IBAN", "계좌"], "zh": ["IBAN", "账户"]}
TRAIN_T = [
    "{ck}: {CARD}, {ek} {EXP}, {vk} {CVV}.",
    "{ck} {CARD} ({ek} {EXP}) {vk}: {CVV}",
    "{ck}: {CARD}\n{ek}: {EXP}\n{vk}: {CVV}",
    "Hi, it's {NAME}. {ck} {CARD}, {vk} {CVV}, {ek} {EXP}. Total {AMT}.",
    '{{"card": "{CARD}", "exp": "{EXP}", "cvc": "{CVV}", "amount": "{AMT}"}}',
    "{ck} {CARD} {vk} {CVV} {ek} {EXP}",
    "{NAME} <{EMAIL}> paid {AMT} at {TIME} with {ck} {CARD}.",
    "{ik}: {IBAN}, BIC {BIC}, {NAME}",
    "Please transfer {AMT} to {ik} {IBAN} (BIC: {BIC}).",
    "{ik} {IBAN}",
    "{vk} {CVV}",
    "{ek} {EXP} / {vk} {CVV}",
]
EVAL_T = [
    "Payment details: {ck} {CARD}; {ek} {EXP}; {vk} {CVV}. Thanks, {NAME}",
    "{NAME} wants to update the {ck} on file to {CARD} (exp {EXP}, {vk} {CVV}).",
    "Refund {AMT} to {ik}: {IBAN}.",
    "order total {AMT} | {ck} {CARD} | {vk} {CVV}",
]


def email(name):
    a = name.lower().replace(" ", ".").replace("'", "")
    return a.encode("ascii", "ignore").decode() + rng.choice(["@gmail.com", "@example.com", "@mail.de", "@outlook.fr"])


def fill(tpl, lang, split):
    kset = "eval" if split == "eval" else "train"
    ck, ek, vk = rng.choice(KW[lang][kset] if kset == "eval" or rng.random() < 0.85 else KW["en"]["train"])
    name = rng.choice(NAMES)
    vals = dict(CARD=(card(), "CREDIT_CARD"), EXP=(expiry(), "DATE"), CVV=(cvv(), "CREDIT_CARD_CVV"), IBAN=(iban(), "BANK_ACCOUNT"),
                BIC=(bic(), "BANK_ACCOUNT"), NAME=(name, "PERSON"), EMAIL=(email(name), "EMAIL"), AMT=(AMOUNT(), None), TIME=(TIME(), None))
    kws = dict(ck=ck, ek=ek, vk=vk, ik=rng.choice(IBAN_KW[lang]))
    text, spans, pos = "", [], 0
    # expand keyword slots first, then value slots left to right, recording spans
    t = tpl.format(**kws, **{k: "\x00" + k + "\x00" for k in vals}) if "{{" not in tpl else \
        tpl.replace("{{", "\x01").replace("}}", "\x02").format(**kws, **{k: "\x00" + k + "\x00" for k in vals}).replace("\x01", "{").replace("\x02", "}")
    parts = t.split("\x00")
    for i, p in enumerate(parts):
        if i % 2 == 0:
            text += p
        else:
            v, lab = vals[p]
            if lab:
                spans.append([len(text), len(text) + len(v), lab])
            text += v
    return text, spans


def make(n, templates, split):
    rows = []
    for _ in range(n):
        lang = rng.choice(list(KW) + ["en"] * 4)
        text, spans = fill(rng.choice(templates), lang, split)
        rows.append(dict(text=text, spans_json=json.dumps(spans), source="synthetic_payment", lang=lang, split=split))
    return rows


os.makedirs(f"{OUT}/eval", exist_ok=True)
tr = pd.read_parquet(f"{V0}/train.parquet"); va = pd.read_parquet(f"{V0}/val.parquet")
syn = pd.DataFrame(make(N, TRAIN_T, "train"))
syn_va = pd.DataFrame(make(300, TRAIN_T, "val"))
ev = pd.DataFrame(make(600, EVAL_T, "eval"))
cols = list(tr.columns)
for d in (syn, syn_va, ev):
    for c in cols:
        if c not in d.columns:
            d[c] = None
pd.concat([tr, syn[cols]]).sample(frac=1.0, random_state=0).to_parquet(f"{OUT}/train.parquet")
pd.concat([va, syn_va[cols]]).to_parquet(f"{OUT}/val.parquet")
ev[cols].to_parquet(f"{OUT}/eval/synthetic_payment.parquet")
for f in os.listdir(f"{V0}/eval"):
    pd.read_parquet(f"{V0}/eval/{f}").to_parquet(f"{OUT}/eval/{f}")
print("v0 train", len(tr), "+ payment", len(syn), "| eval payment", len(ev))
for r in syn.sample(6, random_state=1).itertuples():
    print(repr(r.text), [(r.text[s:e], l) for s, e, l in json.loads(r.spans_json)])
