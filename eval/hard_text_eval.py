"""Score a text CSV with text_dl only and report AUROC, false-alarm and miss rates at the frozen text thresholds, per domain.

    TEXT_DL_MODEL_PATH=models/text_dl_v3 python eval/hard_text_eval.py --csv data/processed/text/hard_only_test.csv --name v3_hard
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd  # noqa: E402
from sklearn.metrics import roc_auc_score  # noqa: E402

from agents.calibration import load_calibration  # noqa: E402
from agents.text_agent.tools import dl_classifier  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--csv", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--n", type=int, default=600)
a = ap.parse_args()
cal = load_calibration()["thresholds"]["text"]
df = pd.read_csv(a.csv)
df = df.groupby("label", group_keys=False).apply(lambda g: g.sample(min(len(g), a.n // 2), random_state=0))
df["s"] = [dl_classifier.score_text(t).score for t in df.text]
out = {"thresholds": cal}
for dom, g in [("ALL", df)] + list(df.groupby("domain")):
    h, m = g[g.label == 0], g[g.label == 1]
    out[dom] = {"n": len(g), "auc": float(roc_auc_score(g.label, g.s)) if g.label.nunique() == 2 else None,
                "human_called_synthetic": float((h.s >= cal["t_hi"]).mean()),
                "human_uncertain": float(((h.s >= cal["t_lo"]) & (h.s < cal["t_hi"])).mean()),
                "ai_called_authentic": float((m.s < cal["t_lo"]).mean()) if len(m) else None}
    print(dom, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out[dom].items()})
Path("eval/results/stress").mkdir(parents=True, exist_ok=True)
Path(f"eval/results/stress/hard_text_{a.name}.json").write_text(json.dumps(out, indent=2))
