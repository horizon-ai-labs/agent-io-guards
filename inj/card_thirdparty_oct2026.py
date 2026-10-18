import pandas as pd,glob,numpy as np
from huggingface_hub import HfApi, hf_hub_download
def roc_auc_score(y,s):
    y=np.asarray(y).astype(bool); r=pd.Series(s).rank().values; n1=y.sum(); n0=len(y)-n1
    return float((r[y].sum()-n1*(n1+1)/2)/(n1*n0))
D=glob.glob('/outputs/eef3c7fe-*')[0]
tr=pd.read_json(hf_hub_download("halilneed/turkish-prompt-injection-benchmark","data/full.jsonl",repo_type="dataset"),lines=True)
fa=pd.read_json(hf_hub_download("halilneed/turkish-prompt-injection-false-alarm-benchmark","data/test.jsonl",repo_type="dataset"),lines=True)
sub=dict(zip(fa.text,fa.subset))
M=[("9baac730","Horizon-Labs/prompt-injection-guard-small (v2.3)","small"),("8f435d9b","Horizon-Labs/prompt-injection-guard-base (v2.2)","base"),("4012d571","Horizon-Labs/prompt-injection-guard-large (v2.1)","large"),
   ("protectai","protectai/deberta-v3-base-prompt-injection-v2",None),("Llama-Prompt-Guard-2-86M","meta-llama/Llama-Prompt-Guard-2-86M",None),("PIGuard","leolee99/PIGuard",None)]
rows=[]
for k,name,sz in M:
    d=glob.glob(D+f'/preds/*{k}*')[0]; t=pd.read_parquet(d+'/tr_pi_full.parquet'); ja=pd.read_parquet(d+'/ja_indirect_doc.parquet')
    y=t.label.values; s=t.score.values; ben=t[y==0]; ti=np.array([sub.get(x)=="task_instruction" for x in tr.text[y==0]])
    rows.append((name,sz,roc_auc_score(y,s),(s[y==1]>=.5).mean(),(ben.score>=.5).mean(),(ben.score.values[ti]>=.5).mean(),(ja.score>=.5).mean()))
tab="\n".join(["| model | Turkish: AUC | Turkish: attacks caught @ 0.5 | Turkish: false alarms @ 0.5 (all benign) | false alarms on assistant data-handling instructions | Japanese indirect injections caught @ 0.5 |","|---|---|---|---|---|---|"]+
              [f"| {'**'+n+'**' if sz else n} | {a:.3f} | {r:.2f} | {f:.3f} | {fi:.2f} | {j:.2f} |" for n,sz,a,r,f,fi,j in rows])
api=HfApi()
for k,name,sz in M[:3]:
    me=[r for r in rows if r[1]==sz][0]
    best=max(rows,key=lambda r:r[2])[1]==sz
    sec=f"""### Third-party multilingual benchmarks (checked October 2026)

Two public benchmarks released after this model was trained, never used for training or tuning:
[halilneed/turkish-prompt-injection-benchmark](https://huggingface.co/datasets/halilneed/turkish-prompt-injection-benchmark) (3,372 Turkish texts:
317 attacks, 3,055 benign, including its false-alarm set of business, banking and assistant-instruction texts) and
[masahiroid/japanese-indirect-prompt-injection-probes](https://huggingface.co/datasets/masahiroid/japanese-indirect-prompt-injection-probes)
(60 Japanese documents with an embedded injection; the document alone is scored).

{tab}

- {"This model ranks Turkish attacks best of the models tested" if best else "Our models rank Turkish attacks better than the other models tested"}, and it catches {me[6]:.0%} of the Japanese indirect injections.
- At the default 0.5 threshold it flags {me[4]:.1%} of the benign Turkish texts. Almost all of these are instructions addressed to the assistant about
  handling data (for example "keep field X, mask all other personal fields"): {me[5]:.0%} of those are flagged, while ordinary business and banking
  text stays under 6%. If your traffic contains many such instructions from trusted users, raise the threshold (for example to 0.9) or
  route them separately.

"""
    R=f"Horizon-Labs/prompt-injection-guard-{sz}"; card=open(hf_hub_download(R,"README.md",force_download=True)).read()
    anchor="## Built-in obfuscation normalizer" if "## Built-in obfuscation normalizer" in card else "## Limitations"
    assert anchor in card and "Third-party multilingual benchmarks" not in card, sz
    card=card.replace(anchor, sec+anchor,1); open(f"/workspace/release/{sz}/README.md","w").write(card)
    print(sz, anchor, api.upload_file(path_or_fileobj=card.encode(), path_in_repo="README.md", repo_id=R, commit_message="Card: third-party Turkish and Japanese prompt-injection benchmarks").oid)
print(tab)
