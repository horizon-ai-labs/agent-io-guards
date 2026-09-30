"""Real-text pool for sentiment training: FineWeb-2 / FineWeb (ODC-BY) snippets in 66 languages + English, to be labelled
by the teacher.   python sentiment/build_pool.py OUT.parquet [PER_LANG]   (CPU job; file list from lid/fw_files.json)
Half the snippets per language come from pages whose URL looks like reviews, forums, comments or blogs (more opinions
than random web text), half from any page. A snippet is 1-3 consecutive sentences (<= 400 chars) or, for 1/4 of them,
a single sentence. Columns: text, lang (FineWeb-2 subset code), src = fw_opinion | fw_any, url.
"""
import json, os, random, re, sys, time
from multiprocessing import Pool

OUT = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 2500
FW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "lid", "fw_files.json")))
LANGS = ("deu_Latn fra_Latn spa_Latn por_Latn ita_Latn arb_Arab arz_Arab hin_Deva cmn_Hani jpn_Jpan kor_Hang rus_Cyrl nld_Latn "
         "pol_Latn ukr_Cyrl ces_Latn slk_Latn bul_Cyrl hrv_Latn srp_Cyrl slv_Latn ron_Latn hun_Latn ell_Grek tur_Latn heb_Hebr "
         "fas_Arab urd_Arab ben_Beng tam_Taml tel_Telu mar_Deva vie_Latn tha_Thai ind_Latn zsm_Latn fil_Latn swh_Latn fin_Latn "
         "swe_Latn nob_Latn dan_Latn cym_Latn eus_Latn cat_Latn mlt_Latn lit_Latn lvs_Latn ekk_Latn uig_Arab arq_Arab ary_Arab "
         "amh_Ethi hau_Latn yor_Latn npi_Deva pan_Guru guj_Gujr kaz_Cyrl azj_Latn kat_Geor hye_Armn als_Latn isl_Latn gle_Latn "
         "bam_Latn").split()
OPINION = re.compile(r"review|forum|comment|blog|opinion|thread|topic|reddit|tripadvisor|trustpilot|yelp|board|discuss|/t/|"
                     r"otzyv|bewertung|avis|resena|recensione|opinie|yorum|kommentar", re.I)
SPLIT = re.compile(r"(?<=[.!?。！？।؟።])\s+|\n+")


def collect(args):
    lang, repo, files, per = args
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    rng = random.Random(sum(map(ord, lang))); fs = HfFileSystem()

    def retry(fn):
        for k in range(6):
            try:
                return fn()
            except Exception:
                if k == 5:
                    raise
                time.sleep(20 * (k + 1))
    op, an, n = [], [], 0
    full = lambda: len(op) >= per // 2 and len(an) >= per - per // 2

    def docs():
        for f in files:
            pf = retry(lambda: pq.ParquetFile(fs.open(f"datasets/{repo}/{f}")))
            for g in range(pf.num_row_groups):
                tb = retry(lambda: pf.read_row_group(g, columns=["text", "url"]))
                yield from zip(tb.column("text").to_pylist(), tb.column("url").to_pylist())
    try:
        for text, url in docs():
            n += 1
            if full() or n > per * 40 or (n > per * 20 and len(op) + len(an) >= per):
                break
            sents = [s.strip() for s in SPLIT.split(text or "") if 10 <= len(s.strip()) <= 400]
            if not sents:
                continue
            i = rng.randrange(len(sents)); k = 1 if rng.random() < 0.25 else rng.choice([1, 2, 3])
            t = " ".join(sents[i:i + k])[:400]
            if OPINION.search(url or ""):
                if len(op) < per // 2:
                    op.append((t, lang, "fw_opinion", url))
            elif len(an) < per - len(op) if n > per * 20 else len(an) < per - per // 2:   # few opinion pages: fill with any
                an.append((t, lang, "fw_any", url))
    except Exception as e:
        print(lang, "failed", e, flush=True)
    rows = op + an
    print(lang, "docs", n, "opinion", len(op), "any", len(an), flush=True)
    return rows


if __name__ == "__main__":
    import pandas as pd
    jobs = [("eng_Latn", "HuggingFaceFW/fineweb", FW["fineweb_eng"], PER * 3)] + \
           [(l, "HuggingFaceFW/fineweb-2", FW["fineweb2"][l], PER) for l in LANGS]
    with Pool(int(os.environ.get("NPROC", 12))) as p:
        res = p.map(collect, jobs)
    df = pd.DataFrame([r for rs in res for r in rs], columns=["text", "lang", "src", "url"]).drop_duplicates("text")
    df.to_parquet(OUT); print("pool", len(df), df.src.value_counts().to_dict(), flush=True)
    os._exit(0)
