"""Upload a trained model + card to the Hub.  python release/publish_model.py SIZE MODEL_DIR EVAL_JSON BASELINES_JSON [ONNX_DIR]"""
import sys, os, shutil, tempfile, json
from huggingface_hub import HfApi, hf_hub_download
size, mdir, ev, bl = sys.argv[1:5]
onnx = sys.argv[5] if len(sys.argv) > 5 else None
BASES = {"small": "jhu-clsp/mmBERT-small", "base": "jhu-clsp/mmBERT-base"}
repo = f"Horizon-Labs/prompt-injection-guard-{size}"
api = HfApi()
api.create_repo(repo, exist_ok=True, private=False)
d = tempfile.mkdtemp(dir="/workspace/.tmp")
for f in ["config.json", "model.safetensors", "tokenizer.json"]:
    os.symlink(f"{mdir}/{f}", f"{d}/{f}")
# transformers-5 saves tokenizer_class=TokenizersBackend, which 4.x cannot load: ship the base model's tokenizer config
for f in ["tokenizer_config.json", "special_tokens_map.json"]:
    shutil.copy(hf_hub_download(BASES[size], f), f"{d}/{f}")
shutil.copy(f"release/{size}/README.md", f"{d}/README.md")
os.makedirs(f"{d}/eval"); shutil.copy(ev, f"{d}/eval/eval_results.json"); shutil.copy(bl, f"{d}/eval/baselines_eval_results.json")
os.makedirs(f"{d}/training")
for f in ["train_log.json", "val_metrics.json"]:
    if os.path.exists(f"{mdir}/{f}"): shutil.copy(f"{mdir}/{f}", f"{d}/training/{f}")
if onnx:
    os.makedirs(f"{d}/onnx")
    for f in os.listdir(onnx): os.symlink(f"{onnx}/{f}", f"{d}/onnx/{f}")
api.upload_folder(folder_path=d, repo_id=repo, commit_message=sys.argv[6] if len(sys.argv) > 6 else "Upload model")
shutil.rmtree(d)
print("https://huggingface.co/" + repo)
