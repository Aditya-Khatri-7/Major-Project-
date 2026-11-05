"""Largest-face area share per image for every cached sample -> eval/cache/face_fraction.json (id -> fraction, 0 = no face found)."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


from agents.image_agent.face_crop import face_fraction  # noqa: E402
from eval.score_agent_images import build_sets  # noqa: E402

out = {}
for which in ("calib", "tests"):
    for name, samples in build_sets(which).items():
        for k, s in enumerate(samples):
            if s["id"] in out:
                continue
            out[s["id"]] = face_fraction(s["path"])
        print(name, len(samples), flush=True)
Path("eval/cache/face_fraction.json").write_text(json.dumps(out))
