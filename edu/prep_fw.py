"""Sample web documents for the multilingual educational-quality classifier (CPU job).
FineWeb-2 (HuggingFaceFW/fineweb-2, ODC-BY) for 65 languages + FineWeb English (HuggingFaceFW/fineweb sample-10BT, ODC-BY): random documents
from the first row groups of each language's first file (file list: lid/fw_files.json), text cut to 6,000 characters.
Per language: TRAIN docs for teacher labelling, and TEST docs (disjoint) for the independently judged multilingual test set.
python edu/prep_fw.py OUT_DIR [TRAIN=3000] [TEST=150] -> OUT_DIR/train_docs.parquet, test_docs.parquet (text, lang)"""
import json, os, random, sys, time
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import pyarrow.parquet as pq
from huggingface_hub import HfFileSystem

OUT = sys.argv[1]; TRAIN = int(os.environ.get("TRAIN", 3000)); TEST = int(os.environ.get("TEST", 150)); os.makedirs(OUT, exist_ok=True)
LANGS = ("rus_Cyrl cmn_Hani deu_Latn jpn_Jpan spa_Latn fra_Latn ita_Latn por_Latn pol_Latn nld_Latn ind_Latn tur_Latn ces_Latn kor_Hang arb_Arab "
         "hun_Latn fas_Arab ron_Latn vie_Latn ukr_Cyrl nob_Latn tha_Thai ell_Grek swe_Latn dan_Latn fin_Latn bul_Cyrl slk_Latn hrv_Latn hin_Deva "
         "lit_Latn heb_Hebr ben_Beng slv_Latn ekk_Latn cat_Latn lvs_Latn zsm_Latn srp_Cyrl tam_Taml urd_Arab mar_Deva tel_Telu swh_Latn kat_Geor "
         "hye_Armn isl_Latn mkd_Cyrl kaz_Cyrl azj_Latn fil_Latn glg_Latn eus_Latn amh_Ethi npi_Deva guj_Gujr kan_Knda mal_Mlym pan_Guru sin_Sinh "
         "uzn_Latn afr_Latn khm_Khmr mya_Mymr yue_Hani").split()
F = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lid", "fw_files.json")))
fs = HfFileSystem()


def retry(fn):
    for k in range(6):
        try:
            return fn()
        except Exception:
            if k == 5:
                raise
            time.sleep(20 * (k + 1))


def collect(lang):
    repo, files = ("HuggingFaceFW/fineweb", F["fineweb_eng"]) if lang == "eng_Latn" else ("HuggingFaceFW/fineweb-2", F["fineweb2"][lang])
    need = (TRAIN * (4 if lang == "eng_Latn" else 1) + TEST) * 3; docs = []
    try:
        pf = retry(lambda: pq.ParquetFile(fs.open(f"datasets/{repo}/{files[0]}")))
        for g in range(pf.num_row_groups):
            docs += [t[:6000] for t in retry(lambda: pf.read_row_group(g, columns=["text"])).column("text").to_pylist() if t and len(t) >= 200]
            if len(docs) >= need:
                break
    except Exception as e:
        print(lang, "FAILED", repr(e)[:200], flush=True); return lang, [], []
    rng = random.Random(lang); rng.shuffle(docs); docs = list(dict.fromkeys(docs))
    ntr = TRAIN * (4 if lang == "eng_Latn" else 1)
    print(lang, len(docs), flush=True)
    return lang, docs[TEST:TEST + ntr], docs[:TEST]


with ThreadPoolExecutor(8) as ex:
    res = list(ex.map(collect, ["eng_Latn"] + LANGS))
tr = pd.DataFrame([dict(text=t, lang=l) for l, a, _ in res for t in a]); te = pd.DataFrame([dict(text=t, lang=l) for l, _, b in res for t in b])
tr.to_parquet(f"{OUT}/train_docs.parquet"); te.to_parquet(f"{OUT}/test_docs.parquet"); print("train", len(tr), "test", len(te), tr.lang.nunique(), "langs")
