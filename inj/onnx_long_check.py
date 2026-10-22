"""int8 vs fp32 ONNX on long inputs with a windowed scan (CPU, onnxruntime), as a user would run it.
python inj/onnx_long_check.py MODEL_DIR PREDS_DIR OUT.json [WINDOW=2048 STRIDE=1792 MIN_TOK=1500 N=400]
Documents: texts from our eval sets (PREDS_DIR/*.parquet, columns text/label) with >= MIN_TOK tokens. Each document is split into
WINDOW-token windows (overlap WINDOW-STRIDE); the document score is the max P(INJECTION) over windows. Reports decision agreement at 0.5
and 0.9, mean / max |p_int8 - p_fp32|, and per-window agreement."""
import glob, json, os, sys, time
import numpy as np, pandas as pd, onnxruntime as ort
from transformers import AutoTokenizer
d, pdir, out = sys.argv[1:4]
W, S, MIN, N = (int(os.environ.get(k, v)) for k, v in (("WINDOW", 2048), ("STRIDE", 1792), ("MIN_TOK", 1500), ("N", 400)))
tok = AutoTokenizer.from_pretrained(d); cfg = json.load(open(f"{d}/config.json")); pos = next(int(k) for k, v in cfg["id2label"].items() if v.upper() == "INJECTION")
docs = []
for p in sorted(glob.glob(f"{pdir}/*.parquet")):
    df = pd.read_parquet(p, columns=["text", "label"]); name = os.path.basename(p)[:-8]
    for t, l in zip(df.text, df.label):
        if len(t) > MIN * 2:
            ids = tok(t, add_special_tokens=False)["input_ids"]
            if len(ids) >= MIN:
                docs.append((name, int(l), ids))
rng = np.random.RandomState(0); rng.shuffle(docs); docs = docs[:N]
so = ort.SessionOptions(); so.intra_op_num_threads = int(os.environ.get("THREADS", 8))
sess = {k: ort.InferenceSession(f"{d}/onnx/{f}", so, providers=["CPUExecutionProvider"]) for k, f in (("fp32", "model.onnx"), ("int8", "model_quantized.onnx"))}
cls, sep = tok.cls_token_id, tok.sep_token_id
res = {"fp32": [], "int8": []}; wins = {"fp32": [], "int8": []}; t0 = time.time()
for name, l, ids in docs:
    for k, s in sess.items():
        ps = []
        for st in range(0, max(1, len(ids) - (W - S)), S):
            w = [cls] + ids[st:st + W - 2] + [sep]
            lo = s.run(None, {"input_ids": np.array([w], np.int64), "attention_mask": np.ones((1, len(w)), np.int64)})[0][0]
            e = np.exp(lo - lo.max()); ps.append(float(e[pos] / e.sum()))
        res[k].append(max(ps)); wins[k] += ps
a, b = np.array(res["fp32"]), np.array(res["int8"]); wa, wb = np.array(wins["fp32"]), np.array(wins["int8"])
L = [len(x[2]) for x in docs]
r = dict(n_docs=len(docs), n_windows=len(wa), tokens_median=int(np.median(L)), tokens_max=int(max(L)), window=W, stride=S,
         doc_agree_05=float(((a >= .5) == (b >= .5)).mean()), doc_agree_09=float(((a >= .9) == (b >= .9)).mean()),
         window_agree_05=float(((wa >= .5) == (wb >= .5)).mean()), mean_abs_diff=float(np.abs(a - b).mean()), max_abs_diff=float(np.abs(a - b).max()),
         n_flagged_fp32=int((a >= .5).sum()), n_injection_docs=int(sum(x[1] for x in docs)), sets=pd.Series([x[0] for x in docs]).value_counts().to_dict(),
         seconds=round(time.time() - t0))
json.dump(r, open(out, "w"), indent=1); print(json.dumps(r, indent=1))
