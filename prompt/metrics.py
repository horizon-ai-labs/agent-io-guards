"""Metrics against reference labels (the independent judge): task accuracy and macro-F1 (argmax over the 11 task columns), Spearman per
complexity dimension, per-language task accuracy. pred / ref: DataFrames with task_* columns (or a 'task' column of names) and dimension columns."""
import numpy as np, pandas as pd
TASKS = ["open_qa", "closed_qa", "summarization", "text_generation", "code_generation", "chatbot", "classification", "rewrite",
         "brainstorming", "extraction", "other"]
DIMS = ["creativity", "reasoning", "contextual_knowledge", "domain_knowledge", "constraints", "complexity"]
rk = lambda a: pd.Series(np.asarray(a, float)).rank().values


def task_of(df):
    return df["task"].values if "task" in df else np.array([TASKS[i] for i in df[[f"task_{t}" for t in TASKS]].values.argmax(1)])


def metrics(pred, ref, lang):
    p, r = task_of(pred), task_of(ref); out = dict(n=len(r), task_acc=float((p == r).mean()))
    f1s = []
    for t in TASKS:
        tp = ((p == t) & (r == t)).sum(); fp = ((p == t) & (r != t)).sum(); fn = ((p != t) & (r == t)).sum()
        if (r == t).sum() >= 10:
            f1s.append(2 * tp / max(1, 2 * tp + fp + fn))
    out["task_macro_f1"] = float(np.mean(f1s))
    lang = np.asarray(lang); out["task_acc_lang_mean"] = float(np.mean([(p[lang == l] == r[lang == l]).mean() for l in np.unique(lang)]))
    out["task_acc_en"] = float((p[lang == "English"] == r[lang == "English"]).mean()) if (lang == "English").any() else None
    for d in DIMS:
        if d in pred:
            ok = ~(pd.isna(pred[d].values) | pd.isna(ref[d].values))
            out[f"spearman_{d}"] = float(np.corrcoef(rk(pred[d].values[ok]), rk(ref[d].values[ok]))[0, 1])
    sp = [v for k, v in out.items() if k.startswith("spearman_")]
    out["spearman_mean"] = float(np.mean(sp)) if sp else None
    return out
