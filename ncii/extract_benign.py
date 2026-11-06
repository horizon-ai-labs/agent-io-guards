"""Real (English) image-editing instructions as benign data for the NCII guard: MagicBrush (CC-BY-4.0, human-written) and OmniEdit-Filtered (MIT).
 Read through the Hugging Face dataset viewer API (/rows returns the text and only URLs for the
images; reading the parquet files would download the images). Rate-limited, with retries. These texts are NOT assumed benign: the
judge labels every text later.  python ncii/extract_benign.py OUT.parquet ROWS_PER_SOURCE [magicbrush,omniedit]"""
import random, sys, time
import pandas as pd, requests

out, per = sys.argv[1], int(sys.argv[2])
API = "https://datasets-server.huggingface.co"
ALL = {"magicbrush": ("osunlp/MagicBrush", "instruction"), "omniedit": ("TIGER-Lab/OmniEdit-Filtered-1.2M", "edited_prompt_list")}
SRC = [ALL[k] for k in (sys.argv[3].split(",") if len(sys.argv) > 3 else ALL)]   # UltraEdit dropped: the viewer API gives no split sizes
rng, rows = random.Random(0), []


def get(url, params):
    for k in range(6):
        r = requests.get(url, params=params, timeout=60)
        if r.status_code == 200:
            return r.json()
        time.sleep(5 * (k + 1))
    raise RuntimeError(f"{url} {params} -> {r.status_code} {r.text[:200]}")


for repo, col in SRC:
    splits = get(f"{API}/splits", {"dataset": repo})["splits"]
    sizes = get(f"{API}/size", {"dataset": repo})["size"]["splits"]
    n_by = {(s["config"], s["split"]): s["num_rows"] for s in sizes}
    got = 0; rng.shuffle(splits); splits = [sp for sp in splits if n_by.get((sp["config"], sp["split"]), 0)][:5]   # a few random splits
    for i, sp in enumerate(splits):
        cfg, split = sp["config"], sp["split"]; n = n_by[(cfg, split)]
        want = (per - got) // (len(splits) - i)
        offsets = list(range(0, n, 100)); rng.shuffle(offsets)
        for off in offsets:
            if want <= 0:
                break
            js = get(f"{API}/rows", {"dataset": repo, "config": cfg, "split": split, "offset": off, "length": 100})
            for r in js["rows"]:
                v = r["row"].get(col)
                for t in (v if isinstance(v, list) else [v]):
                    if isinstance(t, str) and 3 <= len(t.strip()) <= 400:
                        rows.append(dict(text=t.strip(), src=repo.split("/")[1], split=split)); got += 1; want -= 1
            time.sleep(0.5)
    print(repo, got, flush=True)
d = pd.DataFrame(rows).drop_duplicates("text"); d.to_parquet(out, index=False)
print(len(d), d.src.value_counts().to_dict())
