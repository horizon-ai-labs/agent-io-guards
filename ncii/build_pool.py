"""Pool for judging: generated requests (ncii/gen_ncii.py shards) + real benign edit instructions (ncii/extract_benign.py).
Requests from one generator call stay together (group id = run of identical (lang, kind, cat) in a file); test = 10% of groups
(by hash) for generated data, MagicBrush dev+test splits for real edits.  python ncii/build_pool.py OUT.parquet BENIGN.parquet GEN.jsonl ..."""
import hashlib, json, sys
import pandas as pd

out, benign, gens = sys.argv[1], sys.argv[2], sys.argv[3:]
rows = []
for f in gens:
    g, prev = 0, None
    for l in open(f):
        r = json.loads(l)
        key = (r["lang"], r["kind"], r["cat"])
        if key != prev:
            g += 1; prev = key
        if r.get("label") is None:
            continue
        gid = f"{f.rsplit('/', 1)[-1]}:{g}"
        rows.append(dict(text=r["text"], lang=r["lang"], kind=r["kind"], cat=r["cat"], gen_label=r["label"], group=gid, src="generated",
                         split="test" if int(hashlib.md5(gid.encode()).hexdigest(), 16) % 10 == 0 else "train"))
b = pd.read_parquet(benign)
for t, s, sp in zip(b.text, b.src, b.split):
    rows.append(dict(text=t, lang="English", kind="real_edit", cat=s, gen_label=0, group=f"{s}:{t[:40]}", src=s,
                     split="test" if sp in ("dev", "test") else "train"))
d = pd.DataFrame(rows).drop_duplicates("text").reset_index(drop=True); d["id"] = d.index
d.to_parquet(out, index=False)
print(len(d), d.groupby(["src", "split"]).size().to_dict(), d.groupby("kind").gen_label.mean().round(3).to_dict(), "langs", d.lang.nunique())
