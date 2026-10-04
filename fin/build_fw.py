"""Finance-domain web text for financial-sentiment training: FineWeb-2/FineWeb snippets (.gated/emb_texts.parquet, ODC-BY)
whose URL looks financial (finance/business/markets/bank/... in many languages), split into units of 1-3 sentences.
python fin/build_fw.py TEXTS.parquet OUT.parquet   -> text, lang, url
"""
import random, re, sys
import pandas as pd, pyarrow.parquet as pq
src, out = sys.argv[1:3]
RX = re.compile(r"financ|finanz|boerse|börse|borsa|bourse|bolsa|money|invest|stock|econom|business|market|aktie|trading|forex|crypto|"
                r"bank|eastmoney|nikkei|wallstreet|reuters|bloomberg|cnbc|handelsblatt|expansion|lesechos|ekonom|biznes|finans|kabu|"
                r"profit|earnings|wirtschaft|economi|kinh-te|keuangan|iqtisad|eqtesad", re.I)
SPLIT = re.compile(r"(?<=[.!?。！？।؟])\s+")
rng = random.Random(0); rows = []
pf = pq.ParquetFile(src)
for g in range(pf.num_row_groups):
    df = pf.read_row_group(g, columns=["text", "lang", "url"]).to_pandas()
    df = df[df.url.str.contains(RX, na=False)]
    for t, lang, url in df.itertuples(index=False):
        s = [x.strip() for x in SPLIT.split(t) if x.strip()]
        i = 0
        while i < len(s):
            k = rng.choice([1, 1, 2, 3]); u = " ".join(s[i:i + k]); i += k
            if 25 <= len(u) <= 600:
                rows.append((u, lang, url))
d = pd.DataFrame(rows, columns=["text", "lang", "url"]).drop_duplicates("text")
d.to_parquet(out); print(len(d), d.lang.nunique(), d.lang.value_counts().head(8).to_dict())
