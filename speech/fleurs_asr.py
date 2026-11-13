"""Real ASR output for evaluating punctuation / truecasing on noisy speech-recognizer text. FLEURS test (CC-BY-4.0; read FLORES sentences),
one recording per sentence, transcribed by openai/whisper-large-v3-turbo (MIT) with the language given. Downstream, Whisper's text is
lowercased and stripped of punctuation to mimic CTC recognizers (wav2vec2 / MMS style), keeping its word errors.
python speech/fleurs_asr.py OUT.parquet   (GPU; HF_HOME on node-local /tmp)"""
import io, os, struct, sys, tarfile
import numpy as np, pandas as pd, torch
from huggingface_hub import hf_hub_download
from transformers import pipeline

LANGS = {"en_us": ("en", "eng_Latn"), "de_de": ("de", "deu_Latn"), "fr_fr": ("fr", "fra_Latn"), "es_419": ("es", "spa_Latn"), "pt_br": ("pt", "por_Latn"),
         "it_it": ("it", "ita_Latn"), "nl_nl": ("nl", "nld_Latn"), "pl_pl": ("pl", "pol_Latn"), "ru_ru": ("ru", "rus_Cyrl"), "uk_ua": ("uk", "ukr_Cyrl"),
         "tr_tr": ("tr", "tur_Latn"), "ar_eg": ("ar", "arb_Arab"), "hi_in": ("hi", "hin_Deva"), "cmn_hans_cn": ("zh", "zho_Hans"), "ja_jp": ("ja", "jpn_Jpan"),
         "ko_kr": ("ko", "kor_Hang"), "vi_vn": ("vi", "vie_Latn"), "id_id": ("id", "ind_Latn"), "sw_ke": ("sw", "swh_Latn"), "fa_ir": ("fa", "pes_Arab")}
out = sys.argv[1]


def read_wav(b):
    """minimal RIFF reader: PCM int16 (format 1) or IEEE float32 (format 3, used by FLEURS) -> (float32 mono, rate)"""
    pos, fmt, rate, ch, bits, data = 12, None, None, None, None, None
    while pos + 8 <= len(b):
        cid, size = b[pos:pos + 4], struct.unpack("<I", b[pos + 4:pos + 8])[0]
        if cid == b"fmt ":
            fmt, ch, rate = struct.unpack("<HHI", b[pos + 8:pos + 16]); bits = struct.unpack("<H", b[pos + 22:pos + 24])[0]
        elif cid == b"data":
            data = b[pos + 8:pos + 8 + size]
        pos += 8 + size + (size & 1)
    w = 4 if fmt == 3 else 2; data = data[:len(data) - len(data) % w]   # some files have a stray trailing byte
    x = np.frombuffer(data, dtype=np.float32) if fmt == 3 else np.frombuffer(data, dtype=np.int16).astype(np.float32) / 32768.0
    if ch and ch > 1:
        x = x.reshape(-1, ch).mean(1)
    return x.astype(np.float32), rate


asr = pipeline("automatic-speech-recognition", model="openai/whisper-large-v3-turbo", torch_dtype=torch.float16, device=0)
rows = pd.read_parquet(out).to_dict("records") if os.path.exists(out) else []   # resume: languages already transcribed are skipped
done = {r["fleurs"] for r in rows}
for fl, (wl, code) in LANGS.items():
    if fl in done:
        continue
    tsv = pd.read_csv(hf_hub_download("google/fleurs", f"data/{fl}/test.tsv", repo_type="dataset"), sep="\t", header=None, quoting=3,
                      usecols=[0, 1, 2], names=["sid", "file", "raw"]).drop_duplicates("sid")
    want = dict(zip(tsv.file, zip(tsv.sid, tsv.raw)))
    audio, bad = {}, []
    with tarfile.open(hf_hub_download("google/fleurs", f"data/{fl}/audio/test.tar.gz", repo_type="dataset")) as tf:
        for m in tf:
            name = os.path.basename(m.name)
            if m.isfile() and name in want:
                try:
                    audio[name] = read_wav(tf.extractfile(m).read())
                except Exception as e:   # undecodable file: skip and count
                    bad.append(name)
    long_ = [n for n in audio if len(audio[n][0]) > 30 * audio[n][1]]   # Whisper's window is 30 s; skip (do not truncate) longer clips
    names = sorted(n for n in audio if n not in long_)
    res = asr([{"raw": audio[n][0], "sampling_rate": audio[n][1]} for n in names], batch_size=32,
              generate_kwargs={"language": wl, "task": "transcribe"})
    for n, r in zip(names, res):
        sid, raw = want[n]; rows.append(dict(fleurs=fl, lang=code, sid=int(sid), gold_raw=raw, asr_raw=r["text"].strip()))
    print(fl, len(names), "skipped >30s:", len(long_), "undecodable:", len(bad), "e.g.", rows[-1]["gold_raw"][:60], "|", rows[-1]["asr_raw"][:60], flush=True)
    pd.DataFrame(rows).to_parquet(out, index=False)
