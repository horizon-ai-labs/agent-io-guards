"""v1.1 data: REAL-STYLE genuine automated notifications + fraud imitations of the same sector, Qwen3.8-27B via vLLM.
Why: on the third-party SavienWalker/bulwark-bench-tr (real Turkish SMS) v1.0 flags 53% of genuine institutional messages (teacher: 17%).
The v1.0 generator (gen_scam.py) asked for genuine notifications WITHOUT links ("call the number on your card"), so the model learned
"bank/courier message + link = fraud". Real genuine notifications carry links (often shortened), codes, masked numbers and legal footers,
and SMS is often written without diacritics. No benchmark message is shown to the generator; the prompt only describes general traits.
Each prompt: 6 genuine + 4 fraud messages about one sector, one language, one channel. Labels come later from the teacher.
N=... SHARD=i OUT=dir python scam/gen_notif.py   (vLLM venv) -> $OUT/notif_<shard>.jsonl {text, lang, channel, scam, intended}"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from gen_sent import LANGS, parse_json
import importlib.util
_sp = importlib.util.spec_from_file_location("sent_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

CHANNELS = ["SMS", "SMS", "SMS", "push notification text", "email (subject line and body)"]
SECTORS = ["bank account and card alerts (transactions, transfers, statements, limits, overdraft, standing orders)",
           "one-time passcodes and login / device-pairing confirmations", "parcel and courier delivery updates", "online shop order, return and refund updates",
           "mobile operator: data packages, renewals, bills, roaming", "utility and e-invoice notices (electricity, water, gas, internet)",
           "government and e-government services, tax office, municipality", "insurance policies and applications",
           "consumer loans, instalments, credit reports and credit scores", "doctor, hospital and pharmacy appointments and prescriptions",
           "airline, train and bus bookings, check-in and gate changes", "ride-hailing and food delivery apps", "e-wallet and payment apps",
           "customer satisfaction surveys after a purchase or delivery", "privacy / data-protection consent and terms-update notices",
           "streaming, cloud storage and software subscription renewals", "public emergency and civil-protection alerts",
           "school, university and exam notifications", "investment, brokerage and pension accounts", "car services, tolls and parking"]


def prompt(lang, channel, sector):
    return f"""Write 10 different realistic automated {channel} messages in {lang} about: {sector}, as customers in a {lang}-speaking country
actually receive them. Use invented company names, amounts, dates and links (example-style domains), and local details.
- 6 GENUINE messages sent by the real company or agency. Make them look like real automated system messages, which often:
  contain a link to the company's site or a shortened link (to view a statement, track a parcel, see an invoice, rate a delivery,
  read terms); contain a code with "do not share it with anyone, including our staff"; show masked numbers (card ending 1234,
  tracking or account numbers, names like J*** S***); state amounts, due dates or "if this wasn't you, call ..."; end with legal
  footers (company registration number, opt-out instructions such as "reply STOP", sender codes). SMS are terse and, where that is
  common in the country, written without accents or special letters.
- 4 FRAUD messages imitating this same kind of notification (phishing or scam: fake domain, request to enter card data or
  passwords, pay a small fee, or send a received code; urgency or threats), as convincing as real ones, for a security-awareness set.
Vary length, tone and format. Write naturally in {lang}, as a native speaker; do not translate from English. Do not mention these instructions.
Return only JSON: {{"items": [{{"text": "...", "type": "legitimate|fraud"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 32)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(5100 + shard)
    jobs = [(l, rng.choice(CHANNELS), rng.choice(SECTORS)) for l in LANGS for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs],
                    SamplingParams(temperature=0.95, top_p=0.95, max_tokens=3000, seed=191 + shard), chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0
    with open(f"{out}/notif_{shard}.jsonl", "w") as f:
        for (l, c, s), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            for it in (j or {}).get("items", []) if isinstance(j, dict) else []:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 5 <= len(it["text"]) <= 3000:
                    f.write(json.dumps(dict(text=it["text"].strip(), lang=l, channel=c, scam=s, intended=str(it.get("type", ""))),
                                       ensure_ascii=False) + "\n"); k += 1
    print("items", k, "from", len(jobs), "prompts", flush=True); hard_exit(0)
