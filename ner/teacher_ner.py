"""Qwen3.8-27B as an NER teacher (vLLM venv). Entities are returned as JSON surface strings and aligned to character spans.
Eval:  python ner/teacher_ner.py eval OUT.json EVAL_DIR [N_PER_SET]   (text rebuilt from the gold tokens; F1 as ner/eval_ner.py)
Label: IN=passages.parquet SHARD=i NSHARD=n OUT=file.jsonl python ner/teacher_ner.py label
       -> {pid, text, lang, ents: [[start, end, type], ...]} (character offsets, non-overlapping, longest match first)
"""
import glob, json, os, re, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sentiment"))

PROMPT = """Text:
<<<
{text}
>>>

List every named entity in the text, in the order they appear, using these types:
- PER: names of people (real or fictional), including nicknames; not titles or roles alone ("the president").
- ORG: companies, institutions, government bodies, political parties, teams, bands, newspapers, universities.
- LOC: countries, cities, regions, continents, rivers, mountains and other geographic places; streets, buildings,
  airports and other named facilities.
- MISC: other proper names: nationalities and demonyms (German, Brazilian), languages, religions, events, wars,
  products, brands used as products, works of art, books, films, songs, laws, awards.
Copy each entity exactly as written in the text (same spelling, case and script). Do not include common nouns, dates
or numbers. Return only JSON: {{"entities": [{{"text": "...", "type": "PER|ORG|LOC|MISC"}}, ...]}}"""
TYPES = {"PER", "ORG", "LOC", "MISC"}
# v2 (env PROMPT_V=2): CoNLL-style conventions + 3 short examples
PROMPT2 = PROMPT.replace("""Copy each entity exactly as written""", """Conventions: give the name only - no titles (Mr, Dr, President), no leading "the" unless it is part of the name
(The Beatles), no possessive 's. Multi-word names are one entity ("New York Stock Exchange"). Clubs named after a city
are ORG ("Leeds beat Chelsea"). Nationality and language adjectives are MISC ("German", "Brazilian"), but the country
itself is LOC.
Examples:
"Dr. Anna Schmidt joined Siemens in Munich after the 2012 Olympics." -> Anna Schmidt PER; Siemens ORG; Munich LOC; 2012 Olympics MISC
"El Real Madrid fichó al delantero brasileño Vinícius." -> Real Madrid ORG; brasileño MISC; Vinícius PER
"李华在北京大学学习中文。" -> 李华 PER; 北京大学 ORG; 中文 MISC
Copy each entity exactly as written""")
if os.environ.get("PROMPT_V") == "2":
    PROMPT = PROMPT2


def parse_json(t):
    t = re.sub(r"^```(?:json)?|```$", "", t.strip(), flags=re.M).strip()
    s, e = t.find("{"), t.rfind("}")
    try:
        return json.loads(t[s: e + 1])
    except Exception:
        return None


def align(text, ents):
    """entity strings -> non-overlapping char spans; longer strings first; every occurrence is labelled."""
    taken = [False] * len(text); out = []
    for e, ty in sorted({(e["text"].strip(), e["type"]) for e in ents if isinstance(e, dict) and isinstance(e.get("text"), str)
                         and e.get("type") in TYPES and len(e["text"].strip()) >= 1}, key=lambda x: -len(x[0])):
        for m in re.finditer(re.escape(e), text):
            a, b = m.span()
            # avoid matching inside a longer word for space-separated scripts
            if (a > 0 and text[a - 1].isalnum() and text[a].isascii()) or (b < len(text) and text[b].isalnum() and text[b - 1].isascii()):
                continue
            if not any(taken[a:b]):
                out.append([a, b, ty]); taken[a:b] = [True] * (b - a)
    return sorted(out)


def run(llm, texts):
    from vllm import SamplingParams
    outs = llm.chat([[{"role": "user", "content": PROMPT.format(text=t[:3000])}] for t in texts], SamplingParams(temperature=0, max_tokens=800),
                    chat_template_kwargs={"enable_thinking": False}, use_tqdm=False)
    res = []
    for t, o in zip(texts, outs):
        j = parse_json(o.outputs[0].text)
        res.append(align(t, j.get("entities", [])) if isinstance(j, dict) and isinstance(j.get("entities"), list) else None)
    return res


if __name__ == "__main__":
    import pandas as pd
    from vllm import LLM
    from common import hard_exit
    mode = sys.argv[1]
    llm = LLM(os.environ.get("MODEL", "Qwen/Qwen3.8-27B-FP8"), max_model_len=6144, gpu_memory_utilization=0.92, max_num_seqs=512,
              limit_mm_per_prompt={"image": 0, "video": 0})
    if mode == "eval":
        out, ev, n = sys.argv[2], sys.argv[3], int(sys.argv[4]) if len(sys.argv) > 4 else 300
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "eval_ner.py")).read()
        ns = {}; exec(src[src.index("TYPES ="):src.index("res = json")], ns); spans = ns["spans"]
        res = {}
        for p in sorted(glob.glob(f"{ev}/*.parquet")):
            name = os.path.basename(p)[:-8]; df = pd.read_parquet(p).head(n)
            nospace = name.endswith(("_zh", "_ja", "_th"))
            texts, offs = [], []
            for toks in df.tokens:
                o, t = [], ""
                for w in toks:
                    if t and not nospace:
                        t += " "
                    o.append((len(t), len(t) + len(w))); t += w
                texts.append(t); offs.append(o)
            pred = run(llm, texts); allowed = set(df.classes.iloc[0]); tp = fp = fn = 0
            for o, ents, gold_tags in zip(offs, pred, df.tags):
                tags = ["O"] * len(o)
                for a, b, ty in (ents or []):
                    if ty not in allowed:
                        continue
                    first = True
                    for k, (wa, wb) in enumerate(o):
                        if wa < b and wb > a:
                            tags[k] = ("B-" if first else "I-") + ty; first = False
                g, pr = spans(list(gold_tags)), spans(tags)
                tp += len(g & pr); fp += len(pr - g); fn += len(g - pr)
            res[name] = dict(n=len(df), f1=2 * tp / max(1, 2 * tp + fp + fn), parse_fail=sum(e is None for e in pred)); print(name, res[name], flush=True)
        import numpy as np
        for fam in ["conll_", "wikineural_", "wikiann_"]:
            res["_" + fam[:-1]] = float(np.mean([v["f1"] for k, v in res.items() if k.startswith(fam)]))
        print({k: round(v, 4) for k, v in res.items() if k.startswith("_")}, flush=True)
        json.dump(res, open(out, "w"), indent=1)
    else:
        df = pd.read_parquet(os.environ["IN"]); sh, ns_ = int(os.environ.get("SHARD", 0)), int(os.environ.get("NSHARD", 1))
        df = df.iloc[sh::ns_]
        with open(os.environ["OUT"], "w") as f:
            CH = 20000
            for c in range(0, len(df), CH):
                part = df.iloc[c:c + CH]; pred = run(llm, part.text.tolist()); k = 0
                for (pid, text, lang), ents in zip(part[["pid", "text", "lang"]].values, pred):
                    if ents is not None:
                        f.write(json.dumps(dict(pid=int(pid), text=text, lang=lang, ents=ents), ensure_ascii=False) + "\n"); k += 1
                f.flush(); print("labelled", c + len(part), "kept", k, flush=True)
        print("done", flush=True)
    hard_exit(0)
