"""Text pool for embedding distillation: FineWeb-2 / FineWeb (ODC-BY) snippets of 1-8 consecutive sentences (20-2000 chars;
a third are single sentences or short spans), ~90 languages.   python emb/build_texts.py OUT.parquet [PER_LANG]   (CPU job)
Columns: pid, text, lang, url. English gets 4x PER_LANG. Different random seed and documents than rerank/build_passages.py.
"""
import json, os, random, re, sys, time
from multiprocessing import Pool

OUT = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 8000
FW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lid", "fw_files.json")))
LANGS2 = ("afr_Latn als_Latn amh_Ethi hye_Armn azj_Latn eus_Latn bel_Cyrl bos_Latn cym_Latn epo_Latn glg_Latn guj_Gujr hau_Latn "
          "isl_Latn gle_Latn jav_Latn kan_Knda kat_Geor kaz_Cyrl khm_Khmr kir_Cyrl lao_Laoo mal_Mlym mkd_Cyrl mlt_Latn khk_Cyrl mya_Mymr "
          "npi_Deva ory_Orya pan_Guru pbt_Arab sin_Sinh som_Latn sun_Latn tgk_Cyrl tat_Cyrl uig_Arab uzn_Latn xho_Latn zul_Latn "
          "ltz_Latn fao_Latn snd_Arab ckb_Arab kin_Latn").split()
LANGS = ("arb_Arab ben_Beng deu_Latn spa_Latn fas_Arab fin_Latn fra_Latn hin_Deva ind_Latn jpn_Jpan kor_Hang rus_Cyrl swh_Latn "
         "tel_Telu tha_Thai yor_Latn cmn_Hani bul_Cyrl ces_Latn dan_Latn ita_Latn nld_Latn nob_Latn por_Latn ron_Latn srp_Cyrl "
         "swe_Latn pol_Latn tur_Latn ukr_Cyrl vie_Latn heb_Hebr ell_Grek hun_Latn zsm_Latn fil_Latn urd_Arab tam_Taml mar_Deva "
         "slk_Latn hrv_Latn cat_Latn lit_Latn ekk_Latn lvs_Latn slv_Latn").split()
SPLIT = re.compile(r"(?<=[.!?。！？।؟።])\s+|\n+")


def collect(args):
    lang, repo, files, per = args
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    rng = random.Random(1007 + sum(map(ord, lang))); fs = HfFileSystem(); rows = []

    def retry(fn):
        for k in range(6):
            try:
                return fn()
            except Exception:
                if k == 5:
                    raise
                time.sleep(20 * (k + 1))
    try:
        for f in files:
            pf = retry(lambda: pq.ParquetFile(fs.open(f"datasets/{repo}/{f}")))
            for g in range(pf.num_row_groups - 1, -1, -1):   # last row groups first: other documents than the passage pool
                tb = retry(lambda: pf.read_row_group(g, columns=["text", "url"]))
                for text, url in zip(tb.column("text").to_pylist(), tb.column("url").to_pylist()):
                    sents = [s.strip() for s in SPLIT.split(text or "") if len(s.strip()) >= 10]
                    if len(sents) < 2:
                        continue
                    k = rng.choice([1, 1, 2, 3, 4, 6, 8]); i = rng.randrange(max(1, len(sents) - k + 1))
                    t = " ".join(sents[i:i + k])
                    if rng.random() < 0.15 and len(t) > 60:   # short span (query-like)
                        st = rng.randrange(0, len(t) - 40); t = t[st:st + rng.randint(20, 60)]
                    if 20 <= len(t) <= 2000:
                        rows.append((t, lang, url))
                    if len(rows) >= per:
                        break
                if len(rows) >= per:
                    break
            if len(rows) >= per:
                break
    except Exception as e:
        print(lang, "failed", e, flush=True)
    print(lang, len(rows), flush=True)
    return rows


if __name__ == "__main__":
    import pandas as pd
    jobs = [("eng_Latn", "HuggingFaceFW/fineweb", FW["fineweb_eng"], PER * 4)] + [(l, "HuggingFaceFW/fineweb-2", FW["fineweb2"][l], PER) for l in LANGS + [x for x in LANGS2 if x in FW["fineweb2"]]]
    with Pool(int(os.environ.get("NPROC", 14))) as p:
        res = p.map(collect, jobs)
    df = pd.DataFrame([r for rs in res for r in rs], columns=["text", "lang", "url"]).drop_duplicates("text").reset_index(drop=True)
    df.insert(0, "pid", range(len(df)))
    df.to_parquet(OUT); print("passages", len(df), df.lang.nunique(), "langs", flush=True)
    os._exit(0)
