"""Zero-shot release dir: weights + base tokenizer_config (transformers-4.x compatible) + a pipeline smoke test.
python zeroshot/build_release_dir.py SRC BASE OUT"""
import os, shutil, sys
from huggingface_hub import hf_hub_download
src, base, out = sys.argv[1:4]
os.makedirs(out, exist_ok=True)
for f in ["config.json", "model.safetensors", "tokenizer.json", "val_metrics.json", "train_log.json"]:
    if os.path.exists(f"{src}/{f}"):
        shutil.copy(f"{src}/{f}", f"{out}/{f}")
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(base, f), f"{out}/{f}")
from transformers import pipeline
clf = pipeline("zero-shot-classification", model=out, device=0)
for t in ["My card was charged twice for the same order, please refund one.", "Mañana juega el Real Madrid contra el Barça en el Bernabéu.",
          "新しいスマートフォンのバッテリーは一日中持ちます。"]:
    print(clf(t, ["billing", "sports", "technology", "travel", "politics"]))
print(clf("I love the camera but the battery dies by noon.", ["camera", "battery", "screen", "price"], multi_label=True))
