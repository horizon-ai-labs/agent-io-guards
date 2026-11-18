"""Third-party real-message check: SavienWalker/bulwark-bench-tr (CC-BY-4.0; real Turkish SMS/e-mail from one inbox, masked, plus
hand-reviewed hard synthetic contrast pairs). Built by the author of a Turkish fine-tune of our phishing-scam-detector-base, who reports
53% false alarms on real institutional messages. Ambiguous rows excluded (as the dataset card recommends). label: 0 legitimate, 1 spam/fraud.
python scam/build_bulwark_eval.py scam/evals_bulwark   (control plane; tiny download)"""
import os, sys
import pandas as pd
from huggingface_hub import hf_hub_download

out = sys.argv[1]; os.makedirs(out, exist_ok=True)
R = "SavienWalker/bulwark-bench-tr"
rd = lambda c: pd.read_json(hf_hub_download(R, f"data/{c}.jsonl", repo_type="dataset"), lines=True)
sms, em = rd("real_sms"), rd("real_email")
hard = pd.concat([rd("hard_synthetic_v1"), rd("hard_synthetic_v2")], ignore_index=True)
sets = {"bw_sms_test": sms[sms.eval_split == "test"], "bw_sms_dev": sms[sms.eval_split == "dev"], "bw_email": em, "bw_hard": hard}
for k, d in sets.items():
    d = d[~d.get("ambiguous", pd.Series(False, index=d.index)).fillna(False).astype(bool)].copy()
    d["label3"] = d.label; d["label"] = (d.label != "legitimate").astype(int)
    keep = [c for c in ("text", "label", "label3", "category", "difficulty", "pair", "sender_type") if c in d]
    d[keep].reset_index(drop=True).to_parquet(f"{out}/{k}.parquet", index=False)
    print(k, len(d), d.label3.value_counts().to_dict())
