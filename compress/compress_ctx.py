"""Stage A of the compression eval: compress every QA context with every system, through ONE word-level procedure so that systems
differ only in their keep-probabilities. Units = whitespace words (characters for Chinese/Japanese and other unspaced scripts,
compress/align.py). A token-classification compressor gives P(keep) per token (label 1 = keep, as LLMLingua-2); a unit's score is the
mean over its tokens; the context is split into chunks of <= 510 tokens at unit boundaries. The top round(rate * units) units are kept,
in order, with their original separators. QUOTA=para: the rate applies per blank-line paragraph. Systems: none (rate 1), random (seeded), or a model dir / Hub id.
python compress/compress_ctx.py QA.parquet OUT.parquet RATES SYSTEM [SYSTEM ...]   e.g. RATES=0.5,0.33"""
import os, sys
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from align import nospace, units

LANG2 = {"zh": "zho_Hans", "ja": "jpn_Jpan", "th": "tha_Thai"}
QUOTA = os.environ.get("QUOTA", "global")   # global: top-k over the whole context; para: per paragraph


def para_groups(text, U):
    """unit indices grouped by blank-line-separated paragraph."""
    import re
    bounds = [m.end() for m in re.finditer(r"\n\s*\n", text)]; groups = [[] for _ in range(len(bounds) + 1)]
    for k, (a, _) in enumerate(U):
        groups[sum(1 for b in bounds if b <= a)].append(k)
    return [np.array(g) for g in groups if g]


def join_kept(text, U, keep_idx):
    """kept units in order; a space between two kept units only if the original text between them contains whitespace."""
    out, prev_end = [], None
    for k in keep_idx:
        a, b = U[k]
        if prev_end is not None:
            out.append(" " if any(c.isspace() for c in text[prev_end:a]) else "")
        out.append(text[a:b]); prev_end = b
    return "".join(out)


def unit_scores(tok, model, texts_units, device="cuda", bs=32):
    """texts_units: list of (text, U) -> list of np.array scores per unit."""
    import torch
    chunks = []   # (item, unit indices, token ids)
    for it, (text, U) in enumerate(texts_units):
        enc = tok(text, add_special_tokens=False, return_offsets_mapping=True)
        ids, offs = enc["input_ids"], enc["offset_mapping"]
        starts = np.array([a for a, _ in U]); tok_unit = []
        for (a, b) in offs:
            k = int(np.searchsorted(starts, a, side="right")) - 1
            tok_unit.append(k if k >= 0 and a < U[k][1] else -1)
        cur, cur_units, n = [], [], 0
        by_unit = {}
        for t, k in zip(ids, tok_unit):
            by_unit.setdefault(k, []).append(t)
        for k in range(len(U)):
            ts = by_unit.get(k, [])[:510]   # a unit longer than one window (long unspaced phrases) is scored on its first 510 tokens
            if n + len(ts) > 510 and cur:
                chunks.append((it, cur_units, cur)); cur, cur_units, n = [], [], 0
            cur_units.append((k, len(ts))); cur += ts; n += len(ts)
        if cur_units:
            chunks.append((it, cur_units, cur))
    scores = [np.zeros(len(U)) for _, U in texts_units]
    order = sorted(range(len(chunks)), key=lambda i: len(chunks[i][2]))
    cls, sep, pad = tok.cls_token_id, tok.sep_token_id, tok.pad_token_id or 0
    for s in range(0, len(order), bs):
        idx = order[s:s + bs]; L = min(510, max(len(chunks[i][2]) for i in idx)) + 2
        x = torch.full((len(idx), L), pad); m = torch.zeros((len(idx), L), dtype=torch.long)
        for j, i in enumerate(idx):
            seq = [cls] + chunks[i][2][:510] + [sep]; x[j, :len(seq)] = torch.tensor(seq); m[j, :len(seq)] = 1
        with torch.no_grad():
            p = torch.softmax(model(input_ids=x.to(device), attention_mask=m.to(device)).logits.float(), -1)[:, :, 1].cpu().numpy()
        for j, i in enumerate(idx):
            it, cu, _ = chunks[i]; pos = 1
            for k, nt in cu:
                if nt:
                    scores[it][k] = float(p[j, pos:pos + nt].mean()) if pos + nt <= L - 1 else 0.0
                pos += nt
    return scores


if __name__ == "__main__":
    qa_f, out_f, rates = sys.argv[1], sys.argv[2], [float(x) for x in sys.argv[3].split(",")]
    systems = sys.argv[4:]
    qa = pd.read_parquet(qa_f)
    TU = [(c, units(c, nospace(LANG2.get(l, l)))) for c, l in zip(qa.context, qa.lang)]
    rows = []
    for sysname in systems:
        if sysname == "none":
            for q, (c, U) in zip(qa.itertuples(), TU):
                rows.append(dict(qid=q.Index, system="none", rate=1.0, context=c, kept_units=1.0, kept_chars=1.0))
            continue
        if sysname == "random":
            S = [np.random.default_rng(1000 + i).random(len(U)) for i, (c, U) in enumerate(TU)]
            name = "random"
        else:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            tok = AutoTokenizer.from_pretrained(sysname)
            model = AutoModelForTokenClassification.from_pretrained(sysname).cuda().eval()
            S = unit_scores(tok, model, TU); name = sysname.rstrip("/").split("/")[-1] if "/" in sysname else sysname
            if name == "model":
                name = sysname.rstrip("/").split("/")[-2][:8]
            del model; torch.cuda.empty_cache()
        for r in rates:
            for q, (c, U), sc in zip(qa.itertuples(), TU, S):
                if QUOTA == "para":   # each blank-line-separated paragraph keeps its own share (as llmlingua applies the rate per chunk)
                    keep = []
                    for g in para_groups(c, U):
                        n = max(1, int(round(r * len(g))))
                        keep += [g[i] for i in np.argsort(-sc[g], kind="stable")[:n]]
                    keep = sorted(keep); n = len(keep)
                else:
                    n = max(1, int(round(r * len(U))))
                    keep = sorted(np.argsort(-sc, kind="stable")[:n].tolist())
                cc = join_kept(c, U, keep)
                rows.append(dict(qid=q.Index, system=name, rate=r, context=cc, kept_units=n / max(1, len(U)), kept_chars=len(cc) / max(1, len(c))))
        print(sysname, "done", flush=True)
    pd.DataFrame(rows).to_parquet(out_f, index=False)
