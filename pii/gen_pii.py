"""Synthetic PII / secrets documents for the redactor (Qwen3.8-27B via vLLM).

The model writes a realistic document with every sensitive value wrapped as ⟦TYPE:value⟧; we strip the markers and
record character spans. Covers what the public sets lack: secrets in code/config/logs, chat and email threads,
LLM conversations where users paste data, and languages beyond OpenPII's 30. Also asks for a hard-negative variant
with look-alike non-PII (order numbers, version strings, public figures in news, example.com addresses...).
"""
import json, os, random, re, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "gen"))
from gen_v1 import parse_json  # noqa: E402

OUT = os.environ["OUT"]; SHARD = int(os.environ.get("SHARD", 0)); N = int(os.environ.get("N", 200))
MODEL = os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8")
rng = random.Random(7000 + SHARD)

TYPES = ["PERSON", "EMAIL", "PHONE", "STREET_ADDRESS", "LOCATION", "POSTCODE", "DATE", "DATE_OF_BIRTH", "AGE",
         "NATIONAL_ID", "TAX_ID", "PASSPORT", "DRIVER_LICENSE", "CREDIT_CARD", "CREDIT_CARD_CVV", "BANK_ACCOUNT",
         "IP_ADDRESS", "MAC_ADDRESS", "URL", "USERNAME", "PASSWORD", "SECRET", "ACCOUNT_ID", "MEDICAL_ID",
         "VEHICLE_ID", "DEVICE_ID", "ORGANIZATION", "COORDINATE", "LICENSE_NUMBER"]
TYPE_DOC = ("PERSON (names), EMAIL, PHONE, STREET_ADDRESS, LOCATION (city/region/country), POSTCODE, DATE, DATE_OF_BIRTH, AGE, "
            "NATIONAL_ID (SSN, national ID, social insurance numbers), TAX_ID, PASSPORT, DRIVER_LICENSE, CREDIT_CARD, "
            "CREDIT_CARD_CVV, BANK_ACCOUNT (IBAN, account/routing/SWIFT), IP_ADDRESS, MAC_ADDRESS, URL (personal or "
            "identifying URLs), USERNAME, PASSWORD, SECRET (API keys, tokens, private keys, connection strings, cookies), "
            "ACCOUNT_ID (customer/employee/order-holder IDs tied to a person), MEDICAL_ID, VEHICLE_ID (plates, VIN), "
            "DEVICE_ID, ORGANIZATION (employer / company names linked to a person), COORDINATE, LICENSE_NUMBER")
DOCS = ["Slack/Teams thread", "customer support chat transcript", "email thread with signatures", "server log excerpt",
        "application stack trace with request dump", ".env file", "Kubernetes/YAML config with credentials",
        "Python script with hardcoded credentials", "JavaScript/TypeScript code with an API client",
        "SQL dump / INSERT statements", "CSV export of customers", "JSON API response", "git diff that leaks a secret",
        "shell history", "Jupyter notebook cell output", "medical visit note", "HR onboarding form", "invoice",
        "rental application", "bank transfer confirmation", "travel booking confirmation", "police/incident report",
        "school enrollment form", "a user's message to an AI assistant pasting personal data for help",
        "voicemail / call transcript", "CRM record notes", "insurance claim", "court filing excerpt", "resume",
        "WhatsApp group chat", "e-commerce order confirmation", "IT helpdesk ticket", "Terraform/Ansible file",
        "Docker compose file", "browser HAR / curl command with auth headers", "social media DMs"]
LANGS = (["English"] * 10 + ["German", "French", "Spanish", "Portuguese", "Italian", "Dutch", "Polish", "Russian",
         "Ukrainian", "Turkish", "Arabic", "Hindi", "Simplified Chinese", "Japanese", "Korean", "Vietnamese",
         "Indonesian", "Thai", "Czech", "Swedish", "Persian", "Hebrew", "Bengali", "Romanian", "Greek", "Hungarian",
         "Danish", "Finnish", "Swahili", "Tagalog", "Urdu", "Malay", "Tamil", "Serbian", "Bulgarian", "Norwegian"])


def prompt(doc, lang, focus):
    return f"""Generate training data for a PII and secrets redaction model.

Write a realistic {doc} in {lang} (code and config keys may stay in English). 100-400 words, with the
formatting such a document really has. Use invented but realistic values that fit the country and language
(name formats, phone and ID formats, addresses). Emphasize these kinds of values: {focus}.

Wrap EVERY sensitive value in the text as ⟦TYPE:value⟧ using exactly these TYPE names:
{TYPE_DOC}.
Wrap only the value itself, not labels like "Email:". Do not wrap generic words, product names, or public
organizations mentioned without a link to a person. Every occurrence must be wrapped.

Then write a second, different {doc} in {lang} that contains NO personal data and NO secrets, but does contain
things that look similar: order and ticket numbers without owners, version strings, commit hashes, placeholder values
like "YOUR_API_KEY" or example.com addresses, public figures or companies in a news-like context, generic dates of
public events. Write it without any ⟦⟧ markers.

Return only JSON: {{"doc": "...", "negative": "..."}}"""


MARK = re.compile(r"⟦([A-Z_]+):(.*?)⟧", re.S)


def strip_marks(t):
    out, spans, pos = [], [], 0
    for m in MARK.finditer(t):
        out.append(t[pos:m.start()])
        start = sum(len(x) for x in out)
        val = m.group(2)
        out.append(val)
        if m.group(1) in TYPES and val.strip():
            spans.append([start, start + len(val), m.group(1)])
        pos = m.end()
    out.append(t[pos:])
    text = "".join(out)
    return text, spans


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    os.makedirs(OUT, exist_ok=True)
    jobs = []
    for _ in range(N):
        focus = ", ".join(rng.sample(TYPES, 4))
        a = dict(doc=rng.choice(DOCS), lang=rng.choice(LANGS), focus=focus)
        jobs.append((a, prompt(**a)))
    llm = LLM(MODEL, max_model_len=8192, gpu_memory_utilization=0.92, max_num_seqs=512, limit_mm_per_prompt={"image": 0, "video": 0})
    sp = SamplingParams(temperature=0.9, top_p=0.95, max_tokens=3000, seed=300 + SHARD)
    t = time.time()
    outs = llm.chat([[{"role": "user", "content": p}] for _, p in jobs], sp, chat_template_kwargs={"enable_thinking": False})
    print(f"generated {len(outs)} in {time.time()-t:.0f}s", flush=True)
    ok = 0
    with open(f"{OUT}/pii_gen_{SHARD}.jsonl", "w") as f:
        for (a, _), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            rec = dict(args=a, ok=False)
            if isinstance(j, dict) and isinstance(j.get("doc"), str):
                text, spans = strip_marks(j["doc"])
                neg = j.get("negative") if isinstance(j.get("negative"), str) else ""
                if spans and "⟦" not in text and "⟦" not in neg:
                    rec.update(ok=True, text=text, spans=spans, negative=neg)
                    ok += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"ok {ok}/{len(jobs)}", flush=True)
