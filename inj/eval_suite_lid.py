"""Detected language of the synthetic_* configs of Horizon-Labs/prompt-injection-eval-suite (their `lang` column is the language
the generator was ASKED for). Uses our language identifier (int8 ONNX, local release dir; no Hub model download).
python inj/eval_suite_lid.py LID_MODEL_DIR OUT_DIR   -> OUT_DIR/<config>.parquet (+ lang_detected, lang_conf) and summary.json"""
import json, os, sys
import numpy as np, onnxruntime as ort, pandas as pd
from huggingface_hub import hf_hub_download
from tokenizers import Tokenizer

FLORES2ISO = {"eng": "en", "deu": "de", "fra": "fr", "spa": "es", "por": "pt", "ita": "it", "nld": "nl", "pol": "pl", "rus": "ru",
              "ukr": "uk", "tur": "tr", "hin": "hi", "zho": "zh", "jpn": "ja", "kor": "ko", "vie": "vi", "ind": "id", "tha": "th",
              "ces": "cs", "swe": "sv", "pes": "fa", "heb": "he", "ben": "bn", "ron": "ro", "ell": "el", "hun": "hu", "dan": "da",
              "fin": "fi", "swh": "sw", "tgl": "tl"}
md, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
labels = json.load(open(f"{md}/labels.json"))
iso = [("ar" if l.endswith("_Arab") and l[:3] in ("arb", "ars", "arz", "acm", "aeb", "apc", "ary") else FLORES2ISO.get(l[:3], l)) for l in labels]
tok = Tokenizer.from_file(f"{md}/tokenizer.json"); tok.enable_truncation(256); tok.enable_padding()
so = ort.SessionOptions(); so.intra_op_num_threads = 1
sess = ort.InferenceSession(f"{md}/onnx/model_quantized.onnx", so, providers=["CPUExecutionProvider"])
names = [i.name for i in sess.get_inputs()]
summary = {}
for cfg in ("synthetic_direct_test", "synthetic_docs_test", "synthetic_v2_test"):
    d = pd.read_parquet(hf_hub_download("Horizon-Labs/prompt-injection-eval-suite", f"data/{cfg}.parquet", repo_type="dataset"))
    texts = d.text.tolist(); order = np.argsort([len(t) for t in texts]); pred = [None] * len(texts); conf = [0.0] * len(texts)
    for s in range(0, len(texts), 32):
        idx = order[s:s + 32]; enc = tok.encode_batch([texts[i] for i in idx])
        feed = {"input_ids": np.array([e.ids for e in enc], dtype=np.int64),
                "attention_mask": np.array([e.attention_mask for e in enc], dtype=np.int64)}
        logits = sess.run(None, {k: v for k, v in feed.items() if k in names})[0]
        p = np.exp(logits - logits.max(1, keepdims=True)); p /= p.sum(1, keepdims=True)
        for j, i in enumerate(idx):
            pred[i] = iso[int(p[j].argmax())]; conf[i] = round(float(p[j].max()), 4)
    d["lang_detected"] = pred; d["lang_conf"] = conf
    d.to_parquet(f"{out}/{cfg}.parquet", index=False)
    d["match"] = d.lang == d.lang_detected
    per = d.groupby("lang").agg(n=("match", "size"), match=("match", "mean"), detected_en=("lang_detected", lambda x: float((x == "en").mean())))
    summary[cfg] = dict(n=len(d), match=round(float(d.match.mean()), 4),
                        match_non_en=round(float(d[d.lang != "en"].match.mean()), 4),
                        by_label_non_en={int(k): round(float(v), 4) for k, v in d[d.lang != "en"].groupby("label").match.mean().items()},
                        per_lang={k: dict(n=int(r.n), match=round(r.match, 3), detected_en=round(r.detected_en, 3)) for k, r in per.iterrows()})
    print(cfg, {k: v for k, v in summary[cfg].items() if k != "per_lang"}, flush=True)
json.dump(summary, open(f"{out}/summary.json", "w"), indent=1)
