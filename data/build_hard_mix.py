"""Combine hard human/AI text with a replay sample of the original mix. Split by md5(id) so no text appears twice.

    python data/build_hard_mix.py   ->  data/processed/text/hardmix_{train,val,test}.csv and hard_only_test.csv
"""
import glob
import hashlib

import pandas as pd

T = "data/processed/text/"
hard = pd.concat([pd.read_csv(T + "hard_human.csv")] + [pd.read_csv(f) for f in glob.glob(T + "hard_ai_*.csv")]).drop_duplicates("id")


def bucket(i: str) -> str:
    h = int(hashlib.md5(str(i).encode()).hexdigest(), 16) % 100
    return "train" if h < 80 else "val" if h < 90 else "test"


hard["split"] = hard.id.map(bucket)
base = pd.read_csv(T + "mix_train.csv")
replay = pd.concat([base[base.label == k].sample(9000, random_state=3) for k in (0, 1)])
val_base = pd.read_csv(T + "mix_val.csv").groupby("label").sample(1000, random_state=3)
out = {
    "train": pd.concat([hard[hard.split == "train"], replay]),
    "val": pd.concat([hard[hard.split == "val"], val_base]),
    "test": hard[hard.split == "test"],
}
for k, d in out.items():
    name = f"hardmix_{k}.csv" if k != "test" else "hard_only_test.csv"
    d.drop(columns="split", errors="ignore").sample(frac=1, random_state=1).to_csv(T + name, index=False)
    print(k, len(d), d.label.value_counts().to_dict(), d.domain.value_counts().to_dict() if k == "test" else "")
