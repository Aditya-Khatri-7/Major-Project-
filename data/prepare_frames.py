"""Extract frames from deepfake-benchmark videos and crop faces, producing real/ and fake/ image folders.

  FaceForensics++ (c23):
      python data/prepare_frames.py ffpp --root D:/FaceForensics++ --methods Deepfakes FaceSwap NeuralTextures \
          --frames-per-video 10 --max-videos 150 --split-json D:/FaceForensics/dataset/splits/test.json
  Celeb-DF v2 (official test list):
      python data/prepare_frames.py celebdf --root D:/Celeb-DF-v2 --frames-per-video 10

Output: <out-dir>/real/*.jpg, <out-dir>/fake/*.jpg and manifest.csv. Faces are cropped with a 30% margin so the
images look like WildDeepfake's face crops. Face detection: facenet-pytorch MTCNN if installed, else OpenCV Haar.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
from tqdm import tqdm  # noqa: E402


class FaceCropper:
    """Largest-face detector with a margin crop. MTCNN when available, Haar cascade otherwise."""

    def __init__(self, detector: str = "auto", margin: float = 0.3):
        self.margin, self.mtcnn, self.haar = margin, None, None
        if detector in ("auto", "mtcnn"):
            try:
                import torch
                from facenet_pytorch import MTCNN
                self.mtcnn = MTCNN(keep_all=True, device="cuda" if torch.cuda.is_available() else "cpu", post_process=False)
                print("Face detector: MTCNN")
            except ImportError:
                if detector == "mtcnn":
                    raise SystemExit("facenet-pytorch not installed: pip install facenet-pytorch --no-deps")
        if self.mtcnn is None:
            self.haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            print("Face detector: OpenCV Haar cascade (less accurate; install facenet-pytorch --no-deps for MTCNN)")

    def crop(self, frame_bgr: np.ndarray) -> Optional[np.ndarray]:
        h, w = frame_bgr.shape[:2]
        if self.mtcnn is not None:
            boxes, _ = self.mtcnn.detect(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
            if boxes is None or len(boxes) == 0:
                return None
            x1, y1, x2, y2 = max(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        else:
            faces = self.haar.detectMultiScale(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY), 1.1, 5, minSize=(60, 60))
            if len(faces) == 0:
                return None
            x, y, fw, fh = max(faces, key=lambda f: f[2] * f[3])
            x1, y1, x2, y2 = x, y, x + fw, y + fh
        side = max(x2 - x1, y2 - y1) * (1 + 2 * self.margin)
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        left, top = int(max(cx - side / 2, 0)), int(max(cy - side / 2, 0))
        right, bottom = int(min(cx + side / 2, w)), int(min(cy + side / 2, h))
        if right - left < 40 or bottom - top < 40:
            return None
        return frame_bgr[top:bottom, left:right]


def extract_video(path: Path, cropper: FaceCropper, n_frames: int, out_dir: Path, prefix: str) -> list[Path]:
    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    saved: list[Path] = []
    if total <= 0:
        cap.release()
        return saved
    indices = np.linspace(int(total * 0.05), max(int(total * 0.95) - 1, 0), n_frames).astype(int)
    for k, idx in enumerate(indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, frame = cap.read()
        if not ok:
            continue
        face = cropper.crop(frame)
        if face is None:
            continue
        out = out_dir / f"{prefix}_f{k:02d}.jpg"
        cv2.imwrite(str(out), face, [cv2.IMWRITE_JPEG_QUALITY, 95])
        saved.append(out)
    cap.release()
    return saved


def run(jobs: list[tuple[Path, int, str, str]], args) -> None:
    """jobs: (video path, label 1=fake, prefix, method). Processes them and writes the manifest."""
    cropper = FaceCropper(args.detector)
    out = args.out_dir
    (out / "real").mkdir(parents=True, exist_ok=True)
    (out / "fake").mkdir(parents=True, exist_ok=True)
    manifest = []
    for path, label, prefix, method in tqdm(jobs, desc="videos"):
        folder = out / ("fake" if label else "real")
        for img in extract_video(path, cropper, args.frames_per_video, folder, prefix):
            manifest.append({"path": str(img), "label": label, "source_video": path.name, "method": method})
    with open(out / "manifest.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "source_video", "method"])
        w.writeheader()
        w.writerows(manifest)
    n_fake = sum(m["label"] for m in manifest)
    print(f"Saved {len(manifest)} face crops ({n_fake} fake / {len(manifest) - n_fake} real) to {out}")


def cmd_ffpp(args) -> None:
    root, comp = args.root, args.compression
    allowed_real: Optional[set[str]] = None
    allowed_fake: Optional[set[str]] = None
    if args.split_json:
        pairs = json.loads(Path(args.split_json).read_text(encoding="utf-8"))
        allowed_real = {x for pair in pairs for x in pair}
        allowed_fake = {f"{a}_{b}" for a, b in pairs} | {f"{b}_{a}" for a, b in pairs}
    rng = random.Random(args.seed)

    def pick(folder: Path, allowed: Optional[set[str]]) -> list[Path]:
        vids = sorted(folder.glob("*.mp4"))
        if allowed is not None:
            vids = [v for v in vids if v.stem in allowed]
        rng.shuffle(vids)
        return vids[: args.max_videos]

    real_dir = root / "original_sequences" / "youtube" / comp / "videos"
    if not real_dir.exists():
        raise SystemExit(f"Not found: {real_dir}")
    jobs = [(v, 0, f"real_{v.stem}", "original") for v in pick(real_dir, allowed_real)]
    for method in args.methods:
        fake_dir = root / "manipulated_sequences" / method / comp / "videos"
        if not fake_dir.exists():
            raise SystemExit(f"Not found: {fake_dir}")
        jobs += [(v, 1, f"{method}_{v.stem}", method) for v in pick(fake_dir, allowed_fake)]
    print(f"{len(jobs)} videos to process")
    run(jobs, args)


def cmd_celebdf(args) -> None:
    root = args.root
    list_file = args.list_file or root / "List_of_testing_videos.txt"
    if not list_file.exists():
        raise SystemExit(f"Not found: {list_file}")
    jobs = []
    for line in list_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 2:
            continue
        is_real = parts[0] == "1"                       # Celeb-DF list: 1 = real, 0 = synthesised
        video = root / parts[1]
        if video.exists():
            jobs.append((video, 0 if is_real else 1, f"{'real' if is_real else 'fake'}_{video.stem}",
                         "real" if is_real else "celeb-synthesis"))
    random.Random(args.seed).shuffle(jobs)
    if args.max_videos:
        real = [j for j in jobs if j[1] == 0][: args.max_videos]
        fake = [j for j in jobs if j[1] == 1][: args.max_videos]
        jobs = real + fake
    print(f"{len(jobs)} videos to process")
    run(jobs, args)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", type=Path, required=True)
    common.add_argument("--out-dir", type=Path, default=None)
    common.add_argument("--frames-per-video", type=int, default=10)
    common.add_argument("--max-videos", type=int, default=150, help="per class (per method for FF++)")
    common.add_argument("--detector", choices=["auto", "mtcnn", "haar"], default="auto")
    common.add_argument("--seed", type=int, default=42)

    p = sub.add_parser("ffpp", parents=[common])
    p.add_argument("--methods", nargs="+", default=["Deepfakes", "FaceSwap", "NeuralTextures"],
                   help="Deepfakes Face2Face FaceSwap NeuralTextures")
    p.add_argument("--compression", default="c23")
    p.add_argument("--split-json", type=Path, default=None, help="official splits/test.json to keep test videos only")
    p.set_defaults(fn=cmd_ffpp, default_out="data/processed/image/ffpp")

    p = sub.add_parser("celebdf", parents=[common])
    p.add_argument("--list-file", type=Path, default=None)
    p.set_defaults(fn=cmd_celebdf, default_out="data/processed/image/celebdf")

    args = ap.parse_args()
    args.out_dir = args.out_dir or Path(args.default_out)
    args.fn(args)


if __name__ == "__main__":
    main()
