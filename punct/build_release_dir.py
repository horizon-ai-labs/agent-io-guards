"""Assemble a punctuation release dir: weights + base tokenizer_config (transformers-4.x compatible) + punctuate.py, and
check the helper and the deepmultilingualpunctuation package on a few sentences.
python punct/build_release_dir.py SRC BASE_ID OUT"""
import os, shutil, sys
from huggingface_hub import hf_hub_download
src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "train_log.json"]:
    shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
shutil.copy(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "release", "punct", "punctuate.py"), f"{out}/punctuate.py")
sys.path.insert(0, out)
from punctuate import Punctuator
p = Punctuator(out)
for t in ["hello how are you today i am fine thanks and you", "my name is anna i live in berlin what about you",
          "das ist ein test ich hoffe er funktioniert gut", "hola como estas yo estoy bien gracias",
          "今天天气很好我们去公园散步吧你觉得怎么样", "привет как дела у меня все хорошо спасибо"]:
    print("helper |", p(t), flush=True)
try:
    from deepmultilingualpunctuation import PunctuationModel
    m = PunctuationModel(model=out)
    for t in ["hello how are you today i am fine thanks and you", "das ist ein test ich hoffe er funktioniert gut"]:
        print("deepmultilingualpunctuation |", m.restore_punctuation(t), flush=True)
except Exception as e:
    print("deepmultilingualpunctuation FAILED:", repr(e), flush=True)
