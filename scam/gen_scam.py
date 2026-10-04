"""Synthetic multilingual messages for spam / phishing / scam detection, Qwen3.8-27B via vLLM (Apache-2.0 outputs).
Each prompt asks for 10 messages in one language and channel: 4 legitimate (incl. genuine notifications that look similar to
phishing: real one-time codes, delivery updates, bank alerts without links to log in), 2 spam (marketing), 4 fraud (phishing,
scams of a given type). Labels come later from the teacher (scam/teacher_scam.py), not from the generator. Defensive use:
the texts train a detector; no benchmark items are shown to the generator.
N=... SHARD=i OUT=dir python scam/gen_scam.py   (vLLM venv) -> $OUT/gen_<shard>.jsonl {text, lang, channel, scam, intended}
"""
import json, os, random, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))
from gen_sent import LANGS, parse_json
import importlib.util
_sp = importlib.util.spec_from_file_location("sent_common", os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment", "common.py"))
_m = importlib.util.module_from_spec(_sp); _sp.loader.exec_module(_m); hard_exit = _m.hard_exit

CHANNELS = ["SMS", "SMS", "email (subject line and body)", "WhatsApp or Telegram message", "social media direct message",
            "email (subject line and body)", "push notification text", "message in a marketplace app chat"]
SCAMS = ["fake parcel delivery fee", "bank account blocked / verify your account", "tax refund or government benefit", "lottery or prize win",
         "fake job offer or task scam", "investment or crypto scheme", "romance or friendship scam", "'hi mum/dad, new number' family emergency",
         "fake invoice or payment request to a business (CEO fraud)", "account password reset or login alert with a fake link",
         "tech support or virus warning", "utility bill or electricity cut-off", "fake marketplace buyer or seller", "inheritance or advance-fee",
         "toll road or traffic fine", "subscription renewal or streaming account suspended", "charity or disaster donation scam",
         "OTP or verification code theft ('send me the code you received')", "loan approval with upfront fee", "fake customer service refund"]


def prompt(lang, channel, scam):
    return f"""Write 10 different realistic {channel} messages in {lang}, as people in a {lang}-speaking country would receive them.
Use invented names, companies, phone numbers and links (example domains), and local details (currency, banks, couriers, agencies).
- 4 LEGITIMATE: normal personal, work or service messages, including 2 genuine notifications that resemble phishing but are real
  (a one-time code you requested, a delivery update, a bank transaction alert that asks you to call the number on your card,
  an appointment reminder).
- 2 SPAM: unsolicited advertising or promotions (shops, loans, dating, gambling), annoying but not trying to steal anything.
- 4 FRAUD: phishing or scam messages of this type: {scam}. Make them as convincing and varied as real ones (urgency, official tone,
  or friendly informal style, short links), as a security-awareness training set would.
Write naturally in {lang}, as a native speaker; do not translate from English. Do not mention these instructions.
Return only JSON: {{"items": [{{"text": "...", "type": "legitimate|spam|fraud"}}, ...]}}"""


if __name__ == "__main__":
    from vllm import LLM, SamplingParams
    n, shard, out = int(os.environ.get("N", 60)), int(os.environ.get("SHARD", 0)), os.environ["OUT"]
    rng = random.Random(4100 + shard)
    jobs = [(l, rng.choice(CHANNELS), rng.choice(SCAMS)) for l in LANGS for _ in range(n)]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    outs = llm.chat([[{"role": "user", "content": prompt(*j)}] for j in jobs],
                    SamplingParams(temperature=0.95, top_p=0.95, max_tokens=3000, seed=91 + shard), chat_template_kwargs={"enable_thinking": False})
    os.makedirs(out, exist_ok=True); k = 0
    with open(f"{out}/gen_{shard}.jsonl", "w") as f:
        for (l, c, s), o in zip(jobs, outs):
            j = parse_json(o.outputs[0].text)
            for it in (j or {}).get("items", []) if isinstance(j, dict) else []:
                if isinstance(it, dict) and isinstance(it.get("text"), str) and 5 <= len(it["text"]) <= 3000:
                    f.write(json.dumps(dict(text=it["text"].strip(), lang=l, channel=c, scam=s, intended=str(it.get("type", ""))),
                                       ensure_ascii=False) + "\n"); k += 1
    print("items", k, "from", len(jobs), "prompts", flush=True); hard_exit(0)
