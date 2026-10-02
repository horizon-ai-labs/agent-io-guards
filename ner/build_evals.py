"""NER evaluation sets (evaluation only).   python ner/build_evals.py OUT_DIR
Each parquet: tokens (list[str]), tags (list[str], IOB2 with PER/ORG/LOC/MISC), classes (the entity types annotated).
- conll_en: CoNLL-2003 English test (tner/conll2003), PER/ORG/LOC/MISC, all 3,453 sentences.
- wikineural_<lang>: Babelscape/wikineural test (CC-BY-NC-SA, eval only), 9 languages, PER/ORG/LOC/MISC, 1,000 per language.
- wikiann_<lang>: unimelb-nlp/wikiann test, PER/ORG/LOC only, 1,000 per language for 30 languages.
"""
import json, os, sys
import pandas as pd
from huggingface_hub import hf_hub_download

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)


def save(name, toks, tags, classes, cap=1000):
    d = pd.DataFrame({"tokens": toks, "tags": tags}).sample(frac=1.0, random_state=0).head(cap)
    d["classes"] = [classes] * len(d); d.to_parquet(f"{OUT}/{name}.parquet"); print(name, len(d))


lab = json.load(open(hf_hub_download("tner/conll2003", "dataset/label.json", repo_type="dataset"))); i2l = {v: k for k, v in lab.items()}
rows = [json.loads(l) for l in open(hf_hub_download("tner/conll2003", "dataset/test.json", repo_type="dataset"))]
save("conll_en", [r["tokens"] for r in rows], [[i2l[t] for t in r["tags"]] for r in rows], ["PER", "ORG", "LOC", "MISC"], cap=10000)
WN = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC", 7: "B-MISC", 8: "I-MISC"}
for lang in ["de", "en", "es", "fr", "it", "nl", "pl", "pt", "ru"]:
    d = pd.read_parquet(hf_hub_download("Babelscape/wikineural", f"data/test_{lang}-00000-of-00001.parquet", repo_type="dataset"))
    save(f"wikineural_{lang}", d.tokens.tolist(), [[WN[t] for t in ts] for ts in d.ner_tags], ["PER", "ORG", "LOC", "MISC"])
WA = {0: "O", 1: "B-PER", 2: "I-PER", 3: "B-ORG", 4: "I-ORG", 5: "B-LOC", 6: "I-LOC"}
for lang in ["ar", "bg", "bn", "cs", "da", "de", "el", "en", "es", "fa", "fi", "fr", "he", "hi", "hu", "id", "it", "ja", "ko", "nl", "pl",
             "pt", "ro", "ru", "sv", "sw", "th", "tr", "uk", "vi", "zh"]:
    try:
        d = pd.read_parquet(hf_hub_download("unimelb-nlp/wikiann", f"{lang}/test-00000-of-00001.parquet", repo_type="dataset"))
    except Exception as e:
        print(lang, "missing", type(e).__name__); continue
    save(f"wikiann_{lang}", d.tokens.tolist(), [[WA[t] for t in ts] for ts in d.ner_tags], ["PER", "ORG", "LOC"])
