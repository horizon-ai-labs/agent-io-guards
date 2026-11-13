"""Add 'On real speech-recognizer output' to the truecase and punctuation cards (README only) from release/evals/speech_asr_text_eval.json
(speech/eval_asr_text.py, job e7e427a3; FLEURS test via Whisper-large-v3-turbo, speech/fleurs_asr.py, job 5f238d60).
python release/speech_card_section.py [--dry]"""
import json, sys
import numpy as np
from huggingface_hub import HfApi, CommitOperationAdd, hf_hub_download

R = json.load(open("release/evals/speech_asr_text_eval.json"))
UNCASED = {"arb_Arab", "pes_Arab", "kor_Hang", "hin_Deva", "zho_Hans", "jpn_Jpan"}   # scripts without case (+ CJK): casing not meaningful
NAMES = {"eng_Latn": "English", "deu_Latn": "German", "fra_Latn": "French", "spa_Latn": "Spanish", "por_Latn": "Portuguese", "ita_Latn": "Italian",
         "nld_Latn": "Dutch", "pol_Latn": "Polish", "rus_Cyrl": "Russian", "ukr_Cyrl": "Ukrainian", "tur_Latn": "Turkish", "arb_Arab": "Arabic",
         "hin_Deva": "Hindi", "zho_Hans": "Chinese", "jpn_Jpan": "Japanese", "kor_Hang": "Korean", "vie_Latn": "Vietnamese", "ind_Latn": "Indonesian",
         "swh_Latn": "Swahili", "pes_Arab": "Persian"}
REPOS = {"multilingual-punctuation-truecase-small": ("joint_d090dac8", True), "multilingual-punctuation-truecase-base": ("joint_fcf90b17", True),
         "punctuation-restoration-small": ("punct_52506025", False), "punctuation-restoration-base": ("punct_236665fe", False)}
W = R["whisper_native"]["per_lang"]
f3 = lambda x: f"{x:.3f}"


def mean(pl, key, cased_only=False):
    return float(np.mean([v[key] for l, v in pl.items() if not (cased_only and l in UNCASED)]))


def section(key, casing):
    A, C = R[key + "@asr"]["per_lang"], R[key + "@clean"]["per_lang"]
    if casing:
        head = "| input | punctuation F1 (20 languages) | casing F1 (14 languages with cased scripts) |\n|---|---|---|"
        rows = [("this model on the recognizer's text", A), ("this model on the correct text of the same sentences", C), ("Whisper's own punctuation and casing", W)]
        tab = head + "\n" + "\n".join(f"| {n} | {f3(mean(p, 'punct_f1'))} | {f3(mean(p, 'case_f1', True))} |" for n, p in rows)
        per_head = "| language | punctuation: this model | punctuation: Whisper | casing: this model | casing: Whisper |\n|---|---|---|---|---|"
        per = per_head + "\n" + "\n".join(f"| {NAMES[l]} | {f3(A[l]['punct_f1'])} | {f3(W[l]['punct_f1'])} | " +
                                          (f"{f3(A[l]['case_f1'])} | {f3(W[l]['case_f1'])} |" if l not in UNCASED else "n/a | n/a |")
                                          for l in sorted(A, key=lambda l: NAMES[l]))
    else:
        head = "| input | punctuation F1 (20 languages) |\n|---|---|"
        rows = [("this model on the recognizer's text", A), ("this model on the correct text of the same sentences", C), ("Whisper's own punctuation", W)]
        tab = head + "\n" + "\n".join(f"| {n} | {f3(mean(p, 'punct_f1'))} |" for n, p in rows)
        per = "| language | this model | Whisper |\n|---|---|---|\n" + "\n".join(f"| {NAMES[l]} | {f3(A[l]['punct_f1'])} | {f3(W[l]['punct_f1'])} |"
                                                                            for l in sorted(A, key=lambda l: NAMES[l]))
    EU = ["eng_Latn", "deu_Latn", "fra_Latn", "spa_Latn", "por_Latn", "ita_Latn", "nld_Latn", "pol_Latn", "rus_Cyrl", "ukr_Cyrl", "tur_Latn"]
    eu_gap = float(np.mean([W[l]["punct_f1"] - A[l]["punct_f1"] for l in EU]))
    better = [NAMES[l] for l in sorted(A, key=lambda l: W[l]["punct_f1"] - A[l]["punct_f1"]) if A[l]["punct_f1"] - W[l]["punct_f1"] >= .1]
    return f"""### On real speech-recognizer output (checked November 2026)

Speech recognizers make word errors, so we also tested the model on real recognizer output. We used the
[FLEURS](https://huggingface.co/datasets/google/fleurs) test recordings (CC-BY-4.0): 6,424 read sentences in 20 languages. They were
transcribed by [Whisper-large-v3-turbo](https://huggingface.co/openai/whisper-large-v3-turbo), and the transcripts were lowercased and
stripped of punctuation, like the output of recognizers that produce neither. Scoring is end to end: a mark or capital on a word the
recognizer got wrong counts as missed. That is why even perfect punctuation of the recognizer's text cannot reach the scores on the
correct text. Whisper's own punctuation{' and casing' if casing else ''} is shown for comparison.

{tab}

- In the 11 European languages Whisper's own punctuation is slightly better ({eu_gap:+.3f} F1 on average in its favour). This model is
  clearly better in {', '.join(better)}.
  If your recognizer already punctuates well in your language, this model may not add much; if it outputs no punctuation, this is the
  quality to expect.{'''
- Casing is not scored for Arabic, Persian, Hindi, Korean, Chinese and Japanese: their only capitals are Latin-script names, which the
  recognizer usually writes in the native script.''' if casing else ''}
- FLEURS recordings are single read sentences. Long spontaneous speech, with hesitations and false starts, is likely harder. Thirty
  recordings longer than 30 seconds were skipped. Files: `eval/speech_asr_text.json`; scripts: `code/speech/`.

<details>
<summary>Per language, on the recognizer's text</summary>

{per}

</details>

"""


if __name__ == "__main__":
    api = HfApi(); dry = "--dry" in sys.argv
    for repo, (key, casing) in REPOS.items():
        r = f"Horizon-Labs/{repo}"; i = api.model_info(r); t = open(hf_hub_download(r, "README.md", revision=i.sha)).read()
        if "On real speech-recognizer output" in t:
            print(r, "already"); continue
        assert t.count("## Training") == 1, r
        new = t.replace("## Training", section(key, casing) + "## Training")
        if dry:
            print(section(key, casing)[:2500]); break
        ops = [CommitOperationAdd("README.md", new.encode()), CommitOperationAdd("eval/speech_asr_text.json", "release/evals/speech_asr_text_eval.json"),
               CommitOperationAdd("code/speech/fleurs_asr.py", "speech/fleurs_asr.py"), CommitOperationAdd("code/speech/eval_asr_text.py", "speech/eval_asr_text.py")]
        print(r, api.create_commit(r, ops, parent_commit=i.sha, commit_message="Card: punctuation/casing on real speech-recognizer output (FLEURS + Whisper)").oid)
