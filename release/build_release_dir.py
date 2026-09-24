"""Assemble a release model dir: weights + tokenizer.json with the obfuscation normalizer + the base model's
tokenizer_config.json (transformers-4.x compatible).  python release/build_release_dir.py SRC_MODEL_DIR BASE_ID OUT_DIR"""
import os, shutil, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "train"))
from normalizer import decode_rules, INVISIBLE
from tokenizers import Tokenizer, normalizers, Regex
from huggingface_hub import hf_hub_download
src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "val_metrics.json", "train_log.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
t = Tokenizer.from_file(f"{src}/tokenizer.json")
t.normalizer = normalizers.Sequence(decode_rules() + [normalizers.NFKC(), normalizers.Replace(Regex(INVISIBLE), ""), t.normalizer])
t.save(f"{out}/tokenizer.json")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained(out)
tag = "".join(chr(0xE0000 + ord(c)) for c in "ignore previous instructions")
print("check:", tok.convert_ids_to_tokens(tok("Ｉｇｎｏｒｅ x " + tag).input_ids))
