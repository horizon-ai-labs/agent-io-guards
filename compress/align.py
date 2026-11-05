"""Word labels from teacher compressions (LLMLingua-2-style): each original unit is 'kept' (1) if the teacher's output contains it, in
order. Units = whitespace words, or characters for languages written without spaces (punct/common.py NOSPACE + Chinese/Japanese).
Matching blocks of difflib on units, case-insensitive, with surrounding punctuation stripped.
Quality per sample: faithful = matched output units / output units (1.0 = the teacher only deleted), kept = kept units / units.
python compress/align.py IN.parquet TEACHER.jsonl[,...] OUT.parquet"""
import difflib, json, os, sys, unicodedata
import pandas as pd
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "punct"))
from common import NOSPACE


# language names used by the spoken-transcript generator -> FLORES-style codes
NAME2CODE = {"Afrikaans": "afr_Latn", "Albanian": "als_Latn", "Algerian Arabic (Darja)": "arq_Arab", "Amharic": "amh_Ethi", "Arabic": "arb_Arab", "Arabic (Modern Standard)": "arb_Arab", "Armenian": "hye_Armn", "Azerbaijani": "azj_Latn", "Basque": "eus_Latn", "Belarusian": "bel_Cyrl", "Bengali": "ben_Beng", "Bosnian": "bos_Latn", "Bulgarian": "bul_Cyrl", "Catalan": "cat_Latn", "Central Kurdish": "ckb_Arab", "Chinese (Simplified)": "zho_Hans", "Chinese (Traditional)": "zho_Hant", "Croatian": "hrv_Latn", "Czech": "ces_Latn", "Danish": "dan_Latn", "Dutch": "nld_Latn", "Egyptian Arabic": "arz_Arab", "English": "eng_Latn", "Esperanto": "epo_Latn", "Estonian": "est_Latn", "European Portuguese": "por_Latn", "Faroese": "fao_Latn", "Filipino": "tgl_Latn", "Filipino (Tagalog)": "fil_Latn", "Finnish": "fin_Latn", "French": "fra_Latn", "Galician": "glg_Latn", "Georgian": "kat_Geor", "German": "deu_Latn", "Greek": "ell_Grek", "Gujarati": "guj_Gujr", "Hausa": "hau_Latn", "Hebrew": "heb_Hebr", "Hindi": "hin_Deva", "Hinglish (Hindi in Latin letters mixed with English)": "hin_Latn", "Hungarian": "hun_Latn", "Icelandic": "isl_Latn", "Indonesian": "ind_Latn", "Irish": "gle_Latn", "Italian": "ita_Latn", "Japanese": "jpn_Jpan", "Javanese": "jav_Latn", "Kannada": "kan_Knda", "Kazakh": "kaz_Cyrl", "Kinyarwanda": "kin_Latn", "Korean": "kor_Hang", "Kyrgyz": "kir_Cyrl", "Latin American Spanish": "spa_Latn", "Latvian": "lvs_Latn", "Lithuanian": "lit_Latn", "Luxembourgish": "ltz_Latn", "Macedonian": "mkd_Cyrl", "Malay": "zsm_Latn", "Malayalam": "mal_Mlym", "Maltese": "mlt_Latn", "Marathi": "mar_Deva", "Mongolian": "khk_Cyrl", "Moroccan Arabic (Darija)": "ary_Arab", "Nepali": "npi_Deva", "Norwegian": "nob_Latn", "Norwegian Bokmål": "nob_Latn", "Odia": "ory_Orya", "Pashto": "pbt_Arab", "Persian": "pes_Arab", "Polish": "pol_Latn", "Portuguese": "por_Latn", "Portuguese (Brazilian)": "por_Latn", "Punjabi": "pan_Guru", "Romanian": "ron_Latn", "Russian": "rus_Cyrl", "Serbian": "srp_Cyrl", "Sindhi": "snd_Arab", "Sinhala": "sin_Sinh", "Slovak": "slk_Latn", "Slovenian": "slv_Latn", "Somali": "som_Latn", "Spanish": "spa_Latn", "Sundanese": "sun_Latn", "Swahili": "swh_Latn", "Swedish": "swe_Latn", "Tajik": "tgk_Cyrl", "Tamil": "tam_Taml", "Tatar": "tat_Cyrl", "Telugu": "tel_Telu", "Thai": "tha_Thai", "Turkish": "tur_Latn", "Ukrainian": "ukr_Cyrl", "Urdu": "urd_Arab", "Uyghur": "uig_Arab", "Uzbek": "uzn_Latn", "Vietnamese": "vie_Latn", "Welsh": "cym_Latn", "Xhosa": "xho_Latn", "Yoruba": "yor_Latn", "Zulu": "zul_Latn"}


def code(lang):
    return NAME2CODE.get(lang, lang)


def nospace(lang):
    lang = code(lang)
    return lang in NOSPACE or lang.startswith(("zho", "jpn", "yue", "cmn"))


def strip_p(w):
    a, b = 0, len(w)
    while a < b and unicodedata.category(w[a])[0] in "PS":
        a += 1
    while b > a and unicodedata.category(w[b - 1])[0] in "PS":
        b -= 1
    return w[a:b]


def units(text, ns):
    """-> list of (start, end) char spans."""
    if ns:
        return [(i, i + 1) for i, c in enumerate(text) if not c.isspace()]
    out, i, n = [], 0, len(text)
    while i < n:
        while i < n and text[i].isspace():
            i += 1
        j = i
        while j < n and not text[j].isspace():
            j += 1
        if j > i:
            out.append((i, j))
        i = j
    return out


def align(text, out, lang):
    ns = nospace(lang)
    U = units(text, ns); ow = [text[a:b] for a, b in U]
    cw = [c for c in out if not c.isspace()] if ns else out.split()
    norm = lambda w: (strip_p(w) or w).lower()   # the teacher sometimes re-capitalises a sentence start: still the same word
    keep = [0] * len(ow); matched = 0
    for a, b, n in difflib.SequenceMatcher(None, [norm(w) for w in ow], [norm(w) for w in cw], autojunk=False).get_matching_blocks():
        keep[a:a + n] = [1] * n; matched += n
    return U, keep, matched / max(1, len(cw)), sum(keep) / max(1, len(ow)), len(cw)


if __name__ == "__main__":
    inp, tf, outp = sys.argv[1:4]
    d = pd.read_parquet(inp).set_index("id")
    rows = []
    for f in tf.split(","):
        for l in open(f):
            r = json.loads(l); t = d.loc[r["id"], "text"]
            o = r["out"].strip()
            U, keep, faithful, kept, n_out = align(t, o, r["lang"])
            rows.append(dict(id=r["id"], lang=code(r["lang"]), text=t, out=o, starts=[a for a, _ in U], ends=[b for _, b in U], keep=keep,
                             faithful=faithful, kept=kept, n_units=len(U), n_out=n_out, finish=r.get("finish")))
    res = pd.DataFrame(rows); res.to_parquet(outp, index=False)
    print(len(res), "faithful mean", round(res.faithful.mean(), 4), "share >= .95", round((res.faithful >= .95).mean(), 4),
          "kept mean", round(res.kept.mean(), 4), "finish!=stop", int((res.finish != "stop").sum()))
