"""Language identification data: FineWeb-2 / FineWeb (ODC-BY) training text for the FLORES-200 language set, and
FLORES-200 (CC-BY-SA-4.0, evaluation only) dev/devtest evals.   python lid/prep_lid.py OUT [PER_LANG]

Labels are FLORES-200 codes (ISO 639-3 + script), except Chinese: FineWeb-2 has one cmn_Hani subset, so zho_Hans and
zho_Hant are merged into one label zho_Hani. Languages without a FineWeb-2 subset are dropped (listed in stats.json).
Training samples per language: 2/3 "long" (1-3 consecutive sentences, <= 300 chars) and 1/3 "short" (a random span
of 15-60 characters on word boundaries), so the model works on short strings. Any sample containing a FLORES sentence
(normalised exact match) is removed.
OUT/train.parquet, val.parquet (text, label, kind), eval/flores_devtest.parquet (full sentences), eval/flores_short.parquet
(first 20-40 characters of each devtest sentence, cut at a word boundary where possible), labels.json, stats.json
"""
import io, json, os, random, re, sys, tarfile, urllib.request
from multiprocessing import Pool

OUT = sys.argv[1]
PER = int(sys.argv[2]) if len(sys.argv) > 2 else 6000
os.makedirs(f"{OUT}/eval", exist_ok=True)
norm = lambda s: re.sub(r"\s+", " ", s.strip().lower())
SPLIT = re.compile(r"(?<=[.!?。！？।؟።])\s+|\n+")


def flores():
    data = urllib.request.urlopen("https://dl.fbaipublicfiles.com/nllb/flores200_dataset.tar.gz", timeout=600).read()
    tf = tarfile.open(fileobj=io.BytesIO(data))
    out = {"dev": {}, "devtest": {}}
    for m in tf.getmembers():
        parts = [x for x in m.name.split("/") if x not in ("", ".")]   # members look like ./flores200_dataset/devtest/eng_Latn.devtest
        if len(parts) >= 3 and parts[-2] in out and m.isfile() and parts[-1].endswith("." + parts[-2]):
            code = parts[-1].rsplit(".", 1)[0]
            out[parts[-2]][code] = tf.extractfile(m).read().decode("utf-8").splitlines()
    return out


# FLORES code -> FineWeb-2 subset where the names differ (the label stays the FLORES code). Not aliased on purpose:
# prs_Arab (Dari, no subset), ajp_Arab (South Levantine; only North Levantine apc_Arab exists), acq_Arab.
ALIAS = {"est_Latn": "ekk_Latn", "pes_Arab": "fas_Arab", "tgl_Latn": "fil_Latn", "yue_Hant": "yue_Hani", "grn_Latn": "gug_Latn",
         "aka_Latn": "twi_Latn", "kon_Latn": "kng_Latn", "zho_Hans": "cmn_Hani", "zho_Hant": "cmn_Hani"}


# lid/fw_files.json: the first parquet files of every FineWeb-2 subset, listed from the (authenticated) control plane, so
# jobs make no Hub API calls (the cluster's shared IP gets rate-limited for unauthenticated API use).
FW = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "fw_files.json")))


def fw_config(code, configs):
    if code == "eng_Latn":
        return ("HuggingFaceFW/fineweb", FW["fineweb_eng"])
    c = ALIAS.get(code, code)
    return ("HuggingFaceFW/fineweb-2", FW["fineweb2"][c]) if c in configs else None


def label_of(code):
    return "zho_Hani" if code in ("zho_Hans", "zho_Hant") else code


def collect(args):
    label, repo, cfg, bad = args
    import time
    import pyarrow.parquet as pq
    from huggingface_hub import HfFileSystem
    rng = random.Random(sum(map(ord, label)))
    fs = HfFileSystem()

    def retry(fn):
        for k in range(6):
            try:
                return fn()
            except Exception:
                if k == 5:
                    raise
                time.sleep(20 * (k + 1))

    def texts():   # read the text column row group by row group (no datasets schema casting; cheap range reads)
        for f in cfg:
            pf = retry(lambda: pq.ParquetFile(fs.open(f"datasets/{repo}/{f}")))
            for g in range(pf.num_row_groups):
                for t in retry(lambda: pf.read_row_group(g, columns=["text"])).column("text").to_pylist():
                    yield t
    longs, shorts, n_docs = [], [], 0
    try:
        it = texts()
    except Exception as e:
        return label, [], f"load failed: {e}"
    for text in it:
        n_docs += 1
        sents = [s.strip() for s in SPLIT.split(text or "") if len(s.strip()) >= 15]
        if not sents:
            continue
        i = rng.randrange(len(sents)); k = rng.choice([1, 1, 2, 3])
        t = " ".join(sents[i:i + k])[:300]
        if norm(t) not in bad and not any(norm(s) in bad for s in sents[i:i + k]):
            longs.append(t)
        s = sents[rng.randrange(len(sents))]
        L = rng.randint(15, 60)
        if len(s) > L:
            st = rng.randrange(0, len(s) - L + 1)
            sp = s[st:st + L]
            if " " in sp.strip():   # cut to word boundaries for space-separated scripts
                sp = sp[sp.find(" ") + 1:] if st > 0 else sp
                sp = sp[:sp.rfind(" ")] if sp.rfind(" ") > 5 else sp
            s = sp
        if len(s.strip()) >= 5 and norm(s) not in bad:
            shorts.append(s.strip())
        if len(longs) >= PER * 2 // 3 and len(shorts) >= PER // 3 or n_docs >= PER * 3:
            break
    rows = [(t, label, "long") for t in longs[: PER * 2 // 3]] + [(t, label, "short") for t in shorts[: PER // 3]]
    return label, rows, f"docs {n_docs}"


if __name__ == "__main__":
    import pandas as pd
    fl = flores()
    codes = sorted(fl["devtest"])
    bad = {norm(x) for sp in fl.values() for v in sp.values() for x in v}
    configs = set(FW["fineweb2"])
    jobs, dropped = {}, []
    for c in codes:
        fc = fw_config(c, configs)
        if fc is None:
            dropped.append(c)
        else:
            jobs[label_of(c)] = (label_of(c),) + fc + (bad,)
    print("FLORES languages", len(codes), "mapped labels", len(jobs), "dropped", dropped, flush=True)
    with Pool(int(os.environ.get("NPROC", 8))) as p:
        res = p.map(collect, list(jobs.values()))
    rows, stats = [], {}
    for label, r, info in res:
        stats[label] = dict(n=len(r), info=info); rows += r
        print(label, len(r), info, flush=True)
    labels = sorted(l for l, s in stats.items() if s["n"] >= 300)
    df = pd.DataFrame(rows, columns=["text", "label", "kind"])
    df = df[df.label.isin(labels)].drop_duplicates(subset=["text"]).sample(frac=1.0, random_state=0).reset_index(drop=True)
    nv = int(len(df) * 0.02)
    df.iloc[nv:].to_parquet(f"{OUT}/train.parquet"); df.iloc[:nv].to_parquet(f"{OUT}/val.parquet")
    ev = [(s, label_of(c), c) for c in codes for s in fl["devtest"][c]]
    e = pd.DataFrame(ev, columns=["text", "label", "flores_code"])
    e = e[e.label.isin(labels)]   # languages we cannot train on (no FineWeb-2 data) are listed in stats.json instead
    e.to_parquet(f"{OUT}/eval/flores_devtest.parquet")
    rng = random.Random(1)

    def short(s):
        L = rng.randint(20, 40)
        if len(s) <= L:
            return s
        t = s[:L]
        return t[: t.rfind(" ")] if " " in t and t.rfind(" ") >= 10 else t
    e.assign(text=e.text.map(short)).to_parquet(f"{OUT}/eval/flores_short.parquet")
    json.dump(labels, open(f"{OUT}/labels.json", "w"))
    json.dump(dict(stats=stats, dropped_no_fineweb=dropped, dropped_too_few=[l for l, s in stats.items() if s["n"] < 300],
                   n_train=len(df) - nv, n_val=nv), open(f"{OUT}/stats.json", "w"), indent=1)
    print("labels", len(labels), "train", len(df) - nv, "val", nv, "| eval sentences", len(e), flush=True)
    os._exit(0)
