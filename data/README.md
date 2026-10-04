# Data folder

Scripts (committed): `prepare_text.py`, `prepare_frames.py`, `check_datasets.py`.
Data (git-ignored): `data/raw/` (downloads) and `data/processed/` (generated).

```
data/processed/text/   hc3_train|val|calib|test.csv  mage_test.csv  raid_crossgen.csv  raid_adv.csv
data/processed/image/  ffpp/{real,fake}  celebdf/{real,fake}      (+ manifest.csv in each)
D:\archive\            train|valid|test / real|fake                 (WildDeepfake, used in place)
```

Text CSV columns: `id, text, label (0 human / 1 AI), group, source, generator, domain, attack`.
See [../TRAINING_GUIDE.md](../TRAINING_GUIDE.md) steps 1 and 2 for the exact commands.
