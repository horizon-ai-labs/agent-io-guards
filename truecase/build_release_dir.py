"""Assemble a truecase release dir: weights + base tokenizer_config + truecase.py; run the helper on a few sentences and write a
lowercased, mark-free TED sample for the ONNX agreement check.  python truecase/build_release_dir.py SRC BASE_ID OUT EVAL_DIR"""
import os, re, shutil, sys
import pandas as pd
from huggingface_hub import hf_hub_download
src, base, out, ev = sys.argv[1:5]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "train_log.json"]:
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release", "truecase", "truecase.py"), f"{out}/truecase.py")
sys.path.insert(0, out)
from truecase import Truecaser, strip_marks, lower_keep_len
t = Truecaser(out)
for s in ["my name is anna i live in berlin and work for the united nations in new york",
          "hello how are you today i am fine thanks and you", "the nasa mission launched on monday from florida",
          "ich heiße thomas und wohne in münchen was machst du am wochenende", "hola me llamo maría y vivo en madrid con mi familia",
          "je pense que paris est la plus belle ville de france", "привет меня зовут иван я живу в москве",
          "今天天气很好我们去公园散步吧你觉得怎么样", "la reunión con el ministro garcía será el martes en bruselas"]:
    print("helper |", t(s), flush=True)
d = pd.read_parquet(f"{ev}/ted.parquet")
d["text"] = [lower_keep_len(re.sub(r"\s+", " ", strip_marks(x))) for x in d.text]
d.to_parquet(os.path.join(os.path.dirname(out.rstrip("/")), "ted_lower.parquet"), index=False)
