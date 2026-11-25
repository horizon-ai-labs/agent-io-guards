"""Terse machine / acknowledgement text for the injection guard (v2.6). Reported by SamJoshua (Hub discussion #1 on
prompt-injection-guard-small, 2026-11-24): status lines such as "probe q3481 ok n=11", "done", "ack", alone or inside MCP / JSON-RPC
tool results, score 0.86-0.99 INJECTION on small v2.3 and base v2.2 (we reproduced it). Cause: v2.2/v2.3 added very short attack
commands as positives and nothing taught that short, sentence-less text can be harmless machine output.
Templated, no model: status lines, acknowledgements, log/shell lines, HTTP/CLI results, optionally wrapped in tool-result envelopes.
TRAIN and TEST use DISJOINT vocabularies and envelope formats, so the test measures generalisation, not template memorisation.
Contrast positives: short attacks (TRAIN: positives from our training data; TEST: positives from eval sets) placed inside the same
status lines and envelopes, so "envelope or status line = safe" cannot be learned and detection inside envelopes is measured.
python inj/gen_terse.py TRAIN_DATA_DIR EVAL_DIRS(comma) OUT_DIR   -> train_neg / train_pos / test_neg / test_pos .parquet"""
import glob, json, os, random, sys
import pandas as pd

FAM = {
    "train": dict(
        verbs=["probe", "ping", "sync", "build", "deploy", "job", "task", "check", "backup", "upload", "fetch", "index", "migrate", "test",
               "lint", "crawl", "export", "import", "scan", "render", "compile", "publish", "refresh", "warmup", "replicate", "cron"],
        ok=["ok", "OK", "done", "success", "passed", "finished", "complete", "completed", "ready", "succeeded", "green", "up"],
        bad=["failed", "error", "timeout", "skipped", "retrying", "degraded", "pending", "queued", "aborted"],
        acks=["ok", "OK", "done", "ack", "noted", "received", "got it", "thanks", "ok thanks", "sure", "confirmed", "roger", "yes", "no",
              "fine", "all good", "will do", "on it", "k", "kk", "👍", "✅", "listo", "hecho", "erledigt", "fertig", "terminé", "d'accord",
              "готово", "принято", "完成", "好的", "了解", "完了", "확인", "tamam", "pronto", "feito", "gotowe", "klaar", "fatto", "selesai"],
        envs=["mcp", "jsonrpc", "plain", "plain", "plain", "json_kv", "shell"]),
    "test": dict(
        verbs=["heartbeat", "snapshot", "rollout", "ingest", "reindex", "vacuum", "rebalance", "handshake", "poll", "flush", "transcode",
               "provision", "checkpoint", "archive", "prune"],
        ok=["OK", "ok", "done", "succeeded", "healthy", "accepted", "applied", "synced", "alive", "clean"],
        bad=["failed", "unhealthy", "rejected", "stale", "throttled"],
        acks=["copy that", "understood", "noted, thanks", "acknowledged", "affirmative", "sounds good", "perfect",
              "👌", "✔", "vale", "alles klar", "c'est fait", "сделано", "收到", "はい", "네", "oldu", "certo", "jasne", "prima", "beres"],
        envs=["tool_result_xml", "openai_tool", "plain", "plain", "yaml", "log"]),
}


def ident(r):
    return r.choice([f"q{r.randint(100, 99999)}", f"#{r.randint(1, 9999)}", f"{r.choice('abcdefgh')}{r.randint(1, 999):03d}",
                     f"{r.randint(1, 9999)}", "%08x" % r.getrandbits(32), f"run-{r.randint(1, 999)}", ""])


def status(r, f):
    v, i = r.choice(f["verbs"]), ident(r)
    res = r.choice(f["ok"]) if r.random() < 0.8 else r.choice(f["bad"])
    extra = r.choice(["", f"n={r.randint(0, 500)}", f"rows: {r.randint(0, 10**5)}", f"exit={r.choice([0, 0, 0, 1, 2])}",
                      f"{r.randint(1, 9999)}ms", f"({r.uniform(0.1, 99):.1f}s)", f"{r.randint(0, 50)}/{r.randint(50, 60)}",
                      f"code={r.choice([200, 201, 204, 304, 404, 500])}", f"items={r.randint(0, 999)}", f"v{r.randint(1, 9)}.{r.randint(0, 20)}"])
    sep = r.choice([" ", ": ", " - ", " | ", " -> "])
    s = r.choice([f"{v} {i} {res} {extra}", f"{v}{sep}{res} {extra}", f"[{v}] {res} {extra}", f"{v} {res}", f"{i} {res} {extra}",
                  f"{v.upper()} {res.upper()} {extra}", f"{res}: {v} {i} {extra}"])
    return " ".join(s.split())


def terse(r, f):
    k = r.random()
    if k < 0.45:
        return status(r, f)
    if k < 0.75:
        a = r.choice(f["acks"]); return r.choice([a, a.capitalize(), a + ".", a + "!", a.upper()])
    lines = [status(r, f) for _ in range(r.randint(2, 5))]   # multi-line log
    return "\n".join(lines)


def wrap(r, text, env):
    j = lambda o: json.dumps(o, ensure_ascii=False, separators=(",", ":") if r.random() < 0.5 else (", ", ": "))
    if env == "mcp":
        return j({"content": [{"type": "text", "text": text}], "isError": False})
    if env == "jsonrpc":
        return j({"jsonrpc": "2.0", "id": ident(r) or r.randint(1, 999), "result": {"content": [{"type": "text", "text": text}], "isError": False}})
    if env == "json_kv":
        return j({r.choice(["status", "message", "result", "msg", "output"]): text})
    if env == "shell":
        return f"$ {r.choice(['make test', './run.sh', 'npm run build', 'pytest -q', 'kubectl rollout status', 'curl -s localhost/health'])}\n{text}"
    if env == "tool_result_xml":
        return f"<tool_result name=\"{r.choice(['monitor', 'ops', 'db', 'ci'])}\">{text}</tool_result>"
    if env == "openai_tool":
        return j({"role": "tool", "tool_call_id": f"call_{r.getrandbits(40):x}", "content": text})
    if env == "yaml":
        return f"status: {text}" if "\n" not in text else "log: |\n  " + text.replace("\n", "\n  ")
    if env == "log":
        return "\n".join(f"2026-{r.randint(1, 12):02d}-{r.randint(1, 28):02d}T{r.randint(0, 23):02d}:{r.randint(0, 59):02d}Z INFO {l}" for l in text.split("\n"))
    return text


def short_positives(paths, r, n, max_words=14):
    d = pd.concat([pd.read_parquet(p, columns=["text", "label"]) for p in paths], ignore_index=True)
    d = d[(d.label == 1) & (d.text.str.split().str.len() <= max_words) & (d.text.str.len() >= 8)].drop_duplicates("text")
    return d.text.sample(min(n, len(d)), random_state=r.randint(0, 10**6)).tolist()


def make(split, n_neg, attacks, seed):
    r = random.Random(seed); f = FAM[split]; neg, pos = [], []
    for _ in range(n_neg):
        env = r.choice(f["envs"]); neg.append(dict(text=wrap(r, terse(r, f), env), label=0, env=env))
    for a in attacks:   # attack inside a status line and/or an envelope
        env = r.choice(f["envs"]); s = status(r, f)
        t = r.choice([f"{s}. {a}", f"{s}\n{a}", a, f"{a} {s}"])
        pos.append(dict(text=wrap(r, t, env), label=1, env=env))
    return pd.DataFrame(neg).drop_duplicates("text"), pd.DataFrame(pos).drop_duplicates("text")


if __name__ == "__main__":
    tr_dir, ev_dirs, out = sys.argv[1], sys.argv[2].split(","), sys.argv[3]; os.makedirs(out, exist_ok=True)
    r = random.Random(7)
    tr_att = short_positives([f"{tr_dir}/train.parquet"], r, 1500)
    ev_att = short_positives([p for d in ev_dirs for p in glob.glob(f"{d}/*.parquet")], r, 300)
    tn, tp = make("train", 6000, tr_att, 11)
    sn, sp = make("test", 600, ev_att, 12)
    sn, sp = sn[~sn.text.isin(set(tn.text))], sp[~sp.text.isin(set(tp.text))]   # test never repeats a training string
    for name, df in [("train_neg", tn), ("train_pos", tp), ("test_neg", sn), ("test_pos", sp)]:
        df.assign(source=f"terse/{name}", kind=name, lang="mixed").to_parquet(f"{out}/{name}.parquet", index=False)
        print(name, len(df), df.env.value_counts().to_dict())
