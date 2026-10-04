"""Stream the 11 GB RAID train.csv in chunks and keep a sample (full file does not fit in RAM).

All human rows are kept at HUMAN_FRAC, machine rows at `frac`. source_id is kept so we can split by document.
"""
import sys
import pandas as pd
src, dst, frac = sys.argv[1], sys.argv[2], float(sys.argv[3])
HUMAN_FRAC = 0.3
parts = []
cols = ["id", "source_id", "model", "domain", "attack", "generation"]
for i, ch in enumerate(pd.read_csv(src, chunksize=100_000, usecols=cols)):
    h = ch[ch.model == "human"].sample(frac=HUMAN_FRAC, random_state=i)
    m = ch[ch.model != "human"].sample(frac=frac, random_state=i)
    parts += [h, m]
    if i % 20 == 0:
        print("chunk", i, flush=True)
df = pd.concat(parts)
df.to_csv(dst, index=False)
print(len(df), "rows;", (df.model == "human").sum(), "human;", df.source_id.nunique(), "docs")
