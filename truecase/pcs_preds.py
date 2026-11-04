"""Per-word punctuation and casing predictions of 1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase (punctuators package, ONNX)
on the truecase/tc_metrics.py chunks (lowercased, mark-free input). Output words are aligned to the input words (difflib, case-
insensitive, as punct/eval_pcs.py); casing = class of the output word; unaligned words get '0' / lower.  Run in a private venv.
python truecase/pcs_preds.py EVAL_DIR OUT.json [MODEL]"""
import difflib, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tc_metrics import chunks, units
from common_tc import case_of, lower_keep_len
from punctuators.models import PunctCapSegModelONNX
ev, out = sys.argv[1:3]; mname = sys.argv[3] if len(sys.argv) > 3 else "1-800-BAD-CODE/xlm-roberta_punctuation_fullstop_truecase"
items = chunks(ev); m = PunctCapSegModelONNX.from_pretrained(mname); t0 = time.time(); res = []; miss = 0
texts = [lower_keep_len(it["text"]) for it in items]; outs = []
for s in range(0, len(texts), 64):
    outs += m.infer(texts[s:s + 64], apply_sbd=False)
    if s % 6400 == 0:
        print(s, f"{time.time() - t0:.0f}s", flush=True)
for it, o in zip(items, outs):
    ul = "cmn_Hani" if it["lang"].startswith(("zho", "yue")) else it["lang"]
    oc, oe, ol = units(o, ul)   # output text -> marks per unit; the unit text keeps the output's casing
    gw = [lower_keep_len(it["text"][(it["ends"][i - 1] if i else 0):x]).strip() for i, x in enumerate(it["ends"])]
    pw_raw = [oc[(oe[i - 1] if i else 0):x].strip() for i, x in enumerate(oe)]
    p, c = ["0"] * len(gw), [0] * len(gw)
    for tag, a1, a2, b1, b2 in difflib.SequenceMatcher(None, gw, [w.lower() for w in pw_raw], autojunk=False).get_opcodes():
        if tag == "equal":
            p[a1:a2] = ol[b1:b2]; c[a1:a2] = [case_of(w) for w in pw_raw[b1:b2]]
        else:
            miss += a2 - a1
    res.append(dict(p=p, c=c))
json.dump(res, open(out, "w")); print("done", len(res), "unaligned words", miss, f"{time.time() - t0:.0f}s")
