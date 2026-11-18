"""Add 'On a real Turkish inbox' to Horizon-Labs/phishing-scam-detector-base (README + eval json + eval-builder code).
Numbers: job 20963e9c (scam/run_bulwark_check.sh: v1.0 release build 913a6f87 + Qwen teacher on SavienWalker/bulwark-bench-tr, CC-BY-4.0,
ambiguous rows excluded via scam/build_bulwark_eval.py). The v1.1 retraining attempt (97f0e206, 29c489b2) failed its pre-registered gates.
python release/scam/card_bulwark_nov2026.py [--dry]"""
import json, sys
import pandas as pd
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

REPO = "Horizon-Labs/phishing-scam-detector-base"
J = "/outputs/20963e9c-bd97-447e-81ad-cad296460de8"
D = f"{J}/dumps/" + "{}__913a6f87.parquet"
TH = (0.5, 0.7, 0.9, 0.95)
SETS = [("bw_sms_test", "real SMS, test half"), ("bw_sms_dev", "real SMS, development half"), ("bw_email", "real e-mail"),
        ("bw_hard", "hard contrast set (synthetic, reviewed)")]
pc = lambda x: f"{100 * x:.0f}%"
res = {}
for k, _ in SETS:
    d = pd.read_parquet(D.format(k)); leg, unw = d[d.label == 0].p_unsafe, d[d.label == 1].p_unsafe
    res[k] = dict(n_legitimate=len(leg), n_unwanted=len(unw), **{f"fpr@{t}": float((leg >= t).mean()) for t in TH},
                  **{f"recall@{t}": float((unw >= t).mean()) for t in TH})
T = json.load(open(f"{J}/teacher.json"))
res["teacher_qwen3.8-27b"] = {k: dict(fpr=float(T[k]["fpr"]), recall=float(T[k]["recall"]), auc=float(T[k]["auc"])) for k, _ in SETS}
res["source"] = "SavienWalker/bulwark-bench-tr (CC-BY-4.0); ambiguous rows excluded; label 1 = spam or fraud"

tab = "\n".join(["| set | genuine / unwanted | false alarms at 0.5 / 0.7 / 0.9 / 0.95 | unwanted caught at 0.5 / 0.7 / 0.9 / 0.95 |", "|---|---|---|---|"] +
                [f"| {n} | {res[k]['n_legitimate']} / {res[k]['n_unwanted']} | " + " / ".join(pc(res[k][f'fpr@{t}']) for t in TH) + " | " +
                 " / ".join(pc(res[k][f'recall@{t}']) for t in TH) + " |" for k, n in SETS])
tq = res["teacher_qwen3.8-27b"]["bw_sms_test"]
SECTION = f"""### On a real Turkish inbox (third-party benchmark, checked November 2026)

[SavienWalker/bulwark-bench-tr](https://huggingface.co/datasets/SavienWalker/bulwark-bench-tr) (CC-BY-4.0) holds real SMS and e-mail
from one person's Turkish inbox, with personal data masked, plus hand-reviewed contrast pairs of fake and genuine messages. Its
author built it while fine-tuning this model, and found that **it flags about half of the genuine institutional messages** (bank
alerts, one-time codes, parcel tracking, e-invoices, subscription renewals). We reproduced that on the same data (ambiguous rows
excluded, as the dataset recommends):

{tab}

- The false alarms are ordinary automated notifications: they contain links (often shortened), masked card or tracking numbers,
  codes with "do not share", legal footers, and are usually written without Turkish letters (ı, ş, ğ). Our training messages had
  too few genuine notifications of this kind. Our teacher model does much better on the same SMS (false alarms {pc(tq['fpr'])},
  unwanted caught {pc(tq['recall'])}), so the gap is in the training data, not in the label definitions.
- We tried to fix this with about 25,000 new synthetic genuine notifications and accent-free copies, but the retrained model was not
  better on this benchmark (false alarms 51% at 0.5), so it was **not released**. This repository still holds v1.0.
- **For Turkish SMS**, use [SavienWalker/bulwark-scam-detector](https://huggingface.co/SavienWalker/bulwark-scam-detector), a fine-tune
  of this model on real Turkish institutional messages (14% false alarms on the test half, reported by its author). For other
  languages, expect more false alarms on genuine automated notifications than on the smishing sets above: raise the threshold
  (at 0.9 they fall to 15% on the test half, but only 71% of unwanted messages are caught) or allow-list known senders, and measure on your own traffic.
- The benchmark is small (154 test messages) and from one inbox. Numbers: `eval/bulwark_bench_tr.json`; code: `code/scam/build_bulwark_eval.py`.

"""
OLD_LIM = ("- False positives remain (about 11% of normal messages on the smishing test sets at 0.5), especially for genuine\n"
           "  notifications with links and for promotions; tune the threshold on your own traffic.")
NEW_LIM = ("- False positives remain: about 11% of normal messages on the smishing test sets at 0.5, and about half of genuine automated\n"
           "  notifications (bank, parcel, one-time codes) in a real Turkish inbox (see above). Tune the threshold on your own traffic.")

if __name__ == "__main__":
    api = HfApi(); i = api.model_info(REPO); t = open(hf_hub_download(REPO, "README.md", revision=i.sha)).read()
    assert "On a real Turkish inbox" not in t and t.count("## Training") == 1 and t.count(OLD_LIM) == 1
    new = t.replace("## Training", SECTION + "## Training").replace(OLD_LIM, NEW_LIM)
    if "--dry" in sys.argv:
        print(SECTION); print(NEW_LIM); sys.exit()
    json.dump(res, open("release/evals/scam_bulwark_bench_tr.json", "w"), indent=1)
    ops = [CommitOperationAdd("README.md", new.encode()), CommitOperationAdd("eval/bulwark_bench_tr.json", "release/evals/scam_bulwark_bench_tr.json"),
           CommitOperationAdd("code/scam/build_bulwark_eval.py", "scam/build_bulwark_eval.py")]
    print(api.create_commit(REPO, ops, parent_commit=i.sha, commit_message="Card: results on a real Turkish inbox (bulwark-bench-tr)").oid)
