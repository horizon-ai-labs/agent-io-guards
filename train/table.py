import json,sys
res={}
for f in sys.argv[1:]: res.update(json.load(open(f)))
sets=sorted({s for r in res.values() for s in r if not s.startswith('_')})
names=list(res)
short=lambda n:n.split('/')[-1][:18] if not n.startswith('/') else n.split('/')[-2][:8]
print(f"{'set':22s}"+"".join(f"{short(n):>20s}" for n in names))
for s in sets:
    row=[]
    for n in names:
        r=res[n].get(s)
        if not r: row.append("-"); continue
        row.append(f"acc{r['acc']:.3f}" if 'f1' not in r else f"f1 {r['f1']:.3f}/fpr{r['fpr']:.2f}")
    print(f"{s:22s}"+"".join(f"{x:>20s}" for x in row))
