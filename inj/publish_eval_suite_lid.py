"""Upload the synthetic_* configs with lang_detected/lang_conf (inj/eval_suite_lid.py output) + lang_check.json + a card note
to Horizon-Labs/prompt-injection-eval-suite.  python inj/publish_eval_suite_lid.py /tmp/eslid"""
import sys
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download
d = sys.argv[1]; repo = "Horizon-Labs/prompt-injection-eval-suite"; api = HfApi()
card = open(hf_hub_download(repo, "README.md", repo_type="dataset")).read()
anchor = "- Two more sets used on the model card are not included"
note = """- **Language column (checked 2026-11-04).** In the `synthetic_*` configs, `lang` is the language the generator was
  *asked* to write in. Rows were generated directly in that language (no translation step) and had no native-speaker
  review. Our language identifier ([language-detection-small](https://huggingface.co/Horizon-Labs/language-detection-small))
  adds `lang_detected` (ISO 639-1 where mapped, else a FLORES code) and `lang_conf`. Share of non-English rows detected
  in the requested language: `synthetic_docs_test` 92.8%, `synthetic_v2_test` 85.7%, `synthetic_direct_test` 71.8%.
  In the direct set the generator often wrote short messages in English instead (lowest: nl 14%, he/hi/bn 33-34%,
  de 50%). Attack and benign rows are affected about equally (71.4% vs 72.2%), so this does not reveal the label, but
  for per-language scores filter on `lang == lang_detected`. The identifier itself errs too, mostly on JSON- and
  code-like documents. Per-language numbers: `lang_check.json`.
"""
assert anchor in card and "Language column (checked" not in card
card = card.replace(anchor, note + anchor)
ops = [CommitOperationAdd(f"data/{c}.parquet", f"{d}/{c}.parquet") for c in ("synthetic_direct_test", "synthetic_docs_test", "synthetic_v2_test")]
ops += [CommitOperationAdd("lang_check.json", f"{d}/summary.json"), CommitOperationAdd("README.md", card.encode())]
info = api.dataset_info(repo)
c = api.create_commit(repo, ops, repo_type="dataset", parent_commit=info.sha,
                      commit_message="synthetic_* configs: add lang_detected / lang_conf (language-ID check of the requested language) + card note")
print(c.oid)
