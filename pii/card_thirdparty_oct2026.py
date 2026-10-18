"""Add the October 2026 third-party multilingual PII benchmark results (eval job output pii_eval.json) to the pii-redactor cards on the Hub.
python pii/card_thirdparty_oct2026.py PII_EVAL.json"""
import json, sys
import numpy as np
from huggingface_hub import HfApi, hf_hub_download
r = json.load(open(sys.argv[1]))
NAMES = {"10419f82": ("Horizon-Labs/pii-redactor-small (v1.1)", "small"), "296f3e46": ("Horizon-Labs/pii-redactor-base (v1.0)", "base")}
S = [("ar_pii_eval", "Arabic"), ("fr_pii_bench", "French"), ("ko_kdpii", "Korean"), ("tr_kvkk", "Turkish")]
rows = []
for k, v in r.items():
    if "error" in v:
        continue
    name, size = next((NAMES[x] for x in NAMES if x in k), (k, None))
    rows.append((name, size, [v[s]["redact_f1"] for s, _ in S], [v[s]["redact_recall"] for s, _ in S], [v[s]["redact_precision"] for s, _ in S]))
rows.sort(key=lambda x: -np.mean(x[2]))
tab = "\n".join(["| model | mean redaction F1 | " + " | ".join(f"{n} F1 (recall / precision)" for _, n in S) + " |", "|---" * (2 + len(S)) + "|"] +
                [f"| {'**' + n + '**' if sz else '[' + n + '](https://huggingface.co/' + n + ')'} | {np.mean(f):.3f} | " +
                 " | ".join(f"{a:.3f} ({b:.2f} / {c:.2f})" for a, b, c in zip(f, rc, pr)) + " |" for n, sz, f, rc, pr in rows])
api = HfApi()
for size in ("small", "base"):
    me = next(x for x in rows if x[1] == size); rank = [x[1] for x in rows].index(size) + 1
    sec = f"""### Third-party multilingual benchmarks (checked October 2026)

Four public PII benchmarks released after this model was trained, never used for training or tuning:
[Bayanat-Labs/arabic-pii-eval](https://huggingface.co/datasets/Bayanat-Labs/arabic-pii-eval) (1,000 Arabic texts, several dialects),
[JaqueBill/fr-pii-bench](https://huggingface.co/datasets/JaqueBill/fr-pii-bench) (300 French documents),
[1T/veil-pii-ko-stress](https://huggingface.co/datasets/1T/veil-pii-ko-stress) (KDPII Korean dialogue test, 1,000 sampled) and
[newmindai/nm-kvkk-pii-6K](https://huggingface.co/datasets/newmindai/nm-kvkk-pii-6K) (602 Turkish documents). Same label-agnostic redaction
metric as above (`code/pii/evaluate_pii.py`); direct identifiers must be redacted, while company names, dates (Korean), job titles and
medical fields (Turkish) count as optional.

{tab}

- {"This model has the highest mean redaction F1 of the models tested." if rank == 1 else f"This model ranks #{rank} of the models tested on mean redaction F1 (behind our larger version)." if rank == 2 and rows[0][1] else f"This model ranks #{rank} of the models tested on mean redaction F1."}

"""
    R = f"Horizon-Labs/pii-redactor-{size}"; card = open(hf_hub_download(R, "README.md", force_download=True)).read()
    anchor = "### In-distribution held-out sets"
    assert anchor in card and "Third-party multilingual benchmarks" not in card, size
    card = card.replace(anchor, sec + anchor, 1); open(f"release/pii/README_{size}_hub.md", "w").write(card)
    print(size, rank, api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=R, commit_message="Card: third-party Arabic, French, Korean and Turkish PII benchmarks").oid)
print(tab)
