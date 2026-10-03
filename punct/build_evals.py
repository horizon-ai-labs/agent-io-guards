"""Evaluation sets for punctuation restoration (evaluation only, never training).   python punct/build_evals.py OUT_DIR  (CPU job)
- flores: FLORES-200 devtest (CC-BY-SA-4.0): consecutive sentences of the same source article joined into one paragraph
  (written news / travel / wiki text, all languages but those in common.SKIP).
- ted: TED2020 (OPUS; CC-BY-NC-ND-4.0) talk transcripts and their translations: 5 consecutive subtitle sentences, 200
  paragraphs per language (spoken style).
- europarl: Europarl v8 (OPUS): 5 consecutive sentences, 200 per language, the 12 languages of kredor/punctuate-all.
  oliverguhr/fullstop-* and kredor/punctuate-all were trained on Europarl, so this set is in-domain for them.
Output OUT_DIR/{set}.parquet: text (original, punctuated), lang (FLORES-style code), doc.
"""
import io, os, random, sys, tarfile, urllib.request, zipfile
import pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SKIP, units

OUT = sys.argv[1]; os.makedirs(OUT, exist_ok=True)
rng = random.Random(0)


def get(url):
    for k in range(5):
        try:
            return urllib.request.urlopen(url, timeout=900).read()
        except Exception as e:
            print("retry", url, e, flush=True)
    raise RuntimeError(url)


# FLORES-200 devtest
tf = tarfile.open(fileobj=io.BytesIO(get("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz")))
files = {m.name: m for m in tf.getmembers() if m.isfile()}
meta = next(n for n in files if n.endswith("metadata_devtest.tsv"))
md = pd.read_csv(tf.extractfile(files[meta]), sep="\t")
urls = md.URL.tolist()
rows = []
for n in sorted(files):
    if "/devtest/" not in n or not n.endswith(".devtest"):
        continue
    code = os.path.basename(n)[:-8]
    if code in SKIP:
        continue
    sents = tf.extractfile(files[n]).read().decode("utf-8").split("\n")[:len(urls)]
    cur, doc = [], None
    for s, u in zip(sents, urls):
        if u != doc and cur:
            rows.append((("" if code.startswith(("zho", "jpn", "yue")) else " ").join(cur), code, doc)); cur = []
        cur.append(s.strip()); doc = u
    if cur:
        rows.append((" ".join(cur), code, doc))
pd.DataFrame(rows, columns=["text", "lang", "doc"]).to_parquet(f"{OUT}/flores.parquet")
print("flores", len(rows), len({r[1] for r in rows}), flush=True)


def opus(corpus, ver, l2, code, n=200, k=5):
    a, b = sorted([l2, "en"]) if l2 != "en" else ("de", "en")
    z = zipfile.ZipFile(io.BytesIO(get(f"https://object.pouta.csc.fi/OPUS-{corpus}/{ver}/moses/{a}-{b}.txt.zip")))
    name = next(x for x in z.namelist() if x.endswith(f".{a}-{b}.{l2}"))
    lines = z.read(name).decode("utf-8").split("\n")
    out, tries = [], 0
    while len(out) < n and tries < 50 * n:
        tries += 1; i = rng.randrange(0, len(lines) - k)
        par = [x.strip() for x in lines[i:i + k]]
        if any(len(x) < 3 for x in par) or any(units(x, code)[2][-1:] in ([], ["0"]) for x in par):
            continue   # empty lines, headings, lines without final punctuation
        out.append((("" if code.startswith(("zho", "jpn")) else " ").join(par), code, f"{corpus}:{i}"))
    print(corpus, code, len(out), flush=True)
    return out


TED = dict(en="eng_Latn", de="deu_Latn", fr="fra_Latn", es="spa_Latn", it="ita_Latn", pt="por_Latn", nl="nld_Latn", pl="pol_Latn",
           ru="rus_Cyrl", uk="ukr_Cyrl", cs="ces_Latn", ro="ron_Latn", hu="hun_Latn", el="ell_Grek", bg="bul_Cyrl", tr="tur_Latn",
           ar="arb_Arab", he="heb_Hebr", fa="pes_Arab", hi="hin_Deva", id="ind_Latn", vi="vie_Latn", ja="jpn_Jpan", ko="kor_Hang",
           zh_cn="zho_Hans")
EP = dict(en="eng_Latn", de="deu_Latn", fr="fra_Latn", es="spa_Latn", it="ita_Latn", pt="por_Latn", nl="nld_Latn", pl="pol_Latn",
          cs="ces_Latn", bg="bul_Cyrl", sk="slk_Latn", sl="slv_Latn")
for corpus, ver, langs, name in [("TED2020", "v1", TED, "ted"), ("Europarl", "v8", EP, "europarl")]:
    rows = []
    for l2, code in langs.items():
        try:
            rows += opus(corpus, ver, l2, code)
        except Exception as e:
            print("FAILED", corpus, l2, e, flush=True)
    pd.DataFrame(rows, columns=["text", "lang", "doc"]).to_parquet(f"{OUT}/{name}.parquet")
print("done", flush=True)
