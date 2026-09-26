"""Pool 2 for teacher labelling: jailbreak / role-play / over-refusal prompts from our prompt-injection training data v2
(sources already license-checked for the released prompt-injection guard).  python safety/build_pool2.py INJ_V2_DIR OUT.parquet
Why: v1 still ranks WildGuard-style adversarial prompts poorly (PolyGuard prompt AUC .83 vs Qwen3Guard-0.6B .91) and
over-blocks XSTest-style benign prompts (FPR .34). Sources: gen2_framing/* (our Qwen-written jailbreak framings of harmful
requests and benign look-alikes), nvidia_jailbreak, jailbreakv/Template, orbench80k (OR-Bench: benign prompts that look
harmful), awesome_prompts and trustairlab/regular (benign role-play)."""
import sys
import pandas as pd

d = pd.read_parquet(f"{sys.argv[1]}/train.parquet")
keep = d.source.str.match(r"^(gen2_framing/|nvidia_jailbreak$|jailbreakv/Template$|orbench80k$|awesome_prompts$|trustairlab/regular$)")
d = d[keep & d.text.str.len().between(5, 6000)].drop_duplicates(subset=["text"])
out = pd.DataFrame(dict(text=d.text.values, text_pair="", kind="prompt", source="inj/" + d.source.values, lang=d.lang.values))
out.to_parquet(sys.argv[2]); print(len(out)); print(out.source.value_counts())
