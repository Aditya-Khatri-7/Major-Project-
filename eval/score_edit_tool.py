"""Add the edit-localiser verdict (image_edit.jsonl) to every agent score cache, using the same samples as score_agent_images.py."""
import os
import sys
from pathlib import Path

os.environ["SKIP_FORENSICS"] = "1"
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from agents.image_agent.tools import edit_localizer  # noqa: E402
from eval.cache import append_line, done_ids, make_record  # noqa: E402
from eval.score_agent_images import build_sets  # noqa: E402

for which in ("calib", "tests"):
    for name, samples in build_sets(which).items():
        out = Path("eval/cache") / name / "image_edit.jsonl"
        have = done_ids(out)
        for k, s in enumerate(samples):
            if s["id"] not in have:
                append_line(out, make_record(s, edit_localizer.score_image(s["path"])))
        print(name, len(samples), flush=True)
