"""Smoke + fail test against a running API (real models, judges off).

    uvicorn api.main:app --port 8765          # in another terminal, with GEMINI_API_KEY= ANTHROPIC_API_KEY= to keep judges off
    python eval/smoke_api.py --base http://127.0.0.1:8765

Each case states what must happen. A 5xx anywhere is a failure. Exit code 1 if any case fails.
"""
from __future__ import annotations

import argparse
import io
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import httpx
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
RESULTS: list[tuple[str, bool, str]] = []


class PatientClient(httpx.Client):
    """Waits and retries on 429 so functional cases are not confused with the rate-limit case."""

    retry = True

    def request(self, *a, **kw):
        for _ in range(8):
            r = super().request(*a, **kw)
            if r.status_code != 429 or not self.retry:
                return r
            time.sleep(8)
        return r


def api_key() -> str:
    if os.environ.get("API_KEY"):
        return os.environ["API_KEY"]
    env = ROOT / ".env"
    if env.exists():
        for line in env.read_text(encoding="utf-8").splitlines():
            if line.startswith("API_KEY="):
                return line.split("=", 1)[1].strip().strip('"')
    return ""


def check(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, bool(ok), detail))
    print(("PASS " if ok else "FAIL ") + name + (f"  [{detail}]" if detail else ""), flush=True)


def jpeg_bytes(img: Image.Image, **kw) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", **kw)
    return buf.getvalue()


def png_bytes(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def first(pattern_dir: Path, n: int = 1) -> list[Path]:
    files = sorted(p for p in pattern_dir.rglob("*") if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    return files[:n]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8765")
    args = ap.parse_args()
    key = api_key()
    H = {"X-API-Key": key} if key else {}
    c = PatientClient(base_url=args.base, timeout=180, headers=H)

    def post_text(t: str):
        return c.post("/analyze/text", json={"text": t})

    def post_img(data: bytes, name: str = "x.jpg", ctype: str = "image/jpeg"):
        return c.post("/analyze/image", files={"file": (name, data, ctype)})

    # ---------------------------------------------------------------- smoke
    r = c.get("/health")
    check("health 200", r.status_code == 200, r.text[:120])

    demo = ROOT / "demo_samples"
    ai_txt = (demo / "text_AI_generated.txt").read_text(encoding="utf-8")
    hu_txt = (demo / "text_human_written.txt").read_text(encoding="utf-8")
    r = post_text(ai_txt); j = r.json()
    check("demo AI text -> synthetic", r.status_code == 200 and j["verdict"] == "synthetic", f"{j.get('verdict')} {j.get('fused_score')}")
    r = post_text(hu_txt); j = r.json()
    check("demo human text -> authentic", r.status_code == 200 and j["verdict"] == "authentic", f"{j.get('verdict')} {j.get('fused_score')}")
    check("text report has citations/summary", bool(j.get("summary")) and "citations" in j, "")

    r = post_img((demo / "image_deepfake.jpg").read_bytes()); j = r.json()
    check("demo deepfake image -> synthetic", r.status_code == 200 and j["verdict"] == "synthetic", f"{j.get('verdict')} {j.get('fused_score')}")
    jid = j.get("job_id")
    g = c.get(f"/jobs/{jid}/gradcam")
    check("grad-cam served for image job", g.status_code == 200 and g.headers.get("content-type", "").startswith("image"), str(g.status_code))
    r = post_img((demo / "image_real.jpg").read_bytes()); j = r.json()
    check("demo real image -> not synthetic", r.status_code == 200 and j["verdict"] != "synthetic", f"{j.get('verdict')} {j.get('fused_score')}")

    from eval.clip_face_v3 import collect
    wiki, inp, txt = collect("wiki")["test"], collect("inpainting")["test"], collect("text2img")["test"]
    stems = sorted(set(wiki) & set(inp) & set(txt))[:6]
    wrong_edit = wrong_real = 0
    for s in stems:
        jr = post_img(wiki[s].read_bytes()).json()
        je = post_img(inp[s].read_bytes()).json()
        jt = post_img(txt[s].read_bytes()).json()
        wrong_real += jr["verdict"] == "synthetic"
        wrong_edit += je["verdict"] == "authentic"
        wrong_edit += jt["verdict"] == "authentic"
    check("AI-edited / generated faces: never called authentic (12 imgs)", wrong_edit == 0, f"{wrong_edit} wrongly authentic")
    check("real wiki faces: never called synthetic (6 imgs)", wrong_real == 0, f"{wrong_real} wrongly synthetic")

    # full photo = real face on a large background (needs face crop)
    face = Image.open(wiki[stems[0]]).convert("RGB")
    canvas = Image.new("RGB", (1600, 1200), (90, 110, 130))
    canvas.paste(face.resize((400, 400)), (700, 400))
    j = post_img(jpeg_bytes(canvas)).json()
    check("real face inside large photo -> not synthetic", j["verdict"] != "synthetic", f"{j['verdict']} {j['fused_score']}")

    # ---------------------------------------------------------------- fail tests: text
    check("5-word text rejected 400", post_text("too short text here ok").status_code == 400)
    r = post_text(" ".join(["word"] * 30000))
    check("30000-word text rejected 413", r.status_code == 413, str(r.status_code))
    check("empty text rejected 400", post_text("").status_code == 400)
    check("missing field 422", c.post("/analyze/text", json={}).status_code == 422)
    check("non-json body 422", c.post("/analyze/text", content=b"not json", headers={**H, "content-type": "application/json"}).status_code == 422)
    r = post_text(ai_txt.split(".")[0] + ". " + " ".join(["filler"] * 22)); j = r.json()
    ok = r.status_code == 200 and j["verdict"] == "uncertain" and j["escalate_to_human"]
    check("25-word text -> uncertain and escalated", ok, f"{j.get('verdict')}")
    r = post_text("\x00".join(["null"] * 40))
    check("null-byte text does not 5xx", r.status_code < 500, str(r.status_code))
    r = post_text("😀 " * 60)
    check("emoji-only text does not 5xx", r.status_code < 500, str(r.status_code))
    r = post_text("שלום עולם זה טקסט בעברית " * 12)
    check("RTL text does not 5xx", r.status_code < 500, str(r.status_code))
    r = post_text("a" * 50000 + " " + " ".join(["x"] * 30))
    check("one 50k-char word does not 5xx", r.status_code < 500, str(r.status_code))
    r = post_text("<script>alert(1)</script> " * 25 + "'; DROP TABLE jobs; --")
    check("markup/SQL-looking text does not 5xx", r.status_code < 500, str(r.status_code))

    # ---------------------------------------------------------------- fail tests: images
    check("empty image 400", post_img(b"").status_code == 400)
    check("text file named .jpg -> 415", post_img(b"hello world, not an image" * 20).status_code == 415)
    gif = io.BytesIO(); Image.new("RGB", (64, 64), "red").save(gif, format="GIF")
    check("GIF rejected 415", post_img(gif.getvalue(), "a.gif", "image/gif").status_code == 415)
    good = jpeg_bytes(Image.open(demo / "image_real.jpg").convert("RGB"))
    r = post_img(good[: len(good) // 2])
    check("truncated JPEG does not 5xx", r.status_code < 500, str(r.status_code))
    r = post_img(b"\xff\xd8\xff" + os.urandom(5000))
    check("random bytes after JPEG magic do not 5xx", r.status_code < 500, str(r.status_code))
    r = post_img(png_bytes(Image.new("RGBA", (300, 300), (255, 0, 0, 128))), "a.png", "image/png")
    check("RGBA PNG handled", r.status_code == 200, str(r.status_code))
    r = post_img(png_bytes(Image.new("L", (300, 300), 128)), "a.png", "image/png")
    check("grayscale PNG handled", r.status_code == 200, str(r.status_code))
    r = post_img(jpeg_bytes(Image.new("CMYK", (300, 300), (0, 0, 0, 0))))
    check("CMYK JPEG handled", r.status_code < 500, str(r.status_code))
    r = post_img(png_bytes(Image.new("RGB", (1, 1), "white")), "a.png", "image/png")
    check("1x1 image does not 5xx", r.status_code < 500, str(r.status_code))
    r = post_img(png_bytes(Image.new("RGB", (30, 20000), "white")), "a.png", "image/png")
    check("extreme aspect ratio does not 5xx", r.status_code < 500, str(r.status_code))
    bomb = b"\x89PNG\r\n\x1a\n" + struct.pack(">I", 13) + b"IHDR" + struct.pack(">IIBBBBB", 60000, 60000, 8, 2, 0, 0, 0)
    bomb += struct.pack(">I", zlib.crc32(b"IHDR" + struct.pack(">IIBBBBB", 60000, 60000, 8, 2, 0, 0, 0)))
    check("decompression-bomb header rejected (4xx)", 400 <= post_img(bomb, "a.png", "image/png").status_code < 500)
    big = png_bytes(Image.frombytes("RGB", (3000, 3000), os.urandom(3000 * 3000 * 3)))
    r = post_img(big, "a.png", "image/png")
    check("oversize (> limit) image rejected 413", r.status_code == 413, f"{len(big) // 2**20} MB -> {r.status_code}")
    r = post_img(jpeg_bytes(Image.open(demo / "image_real.jpg").convert("RGB")), "../../evil.jpg")
    check("path-traversal filename harmless", r.status_code < 500, str(r.status_code))

    # ---------------------------------------------------------------- fail tests: jobs, auth, misc
    check("unknown job 404", c.get("/jobs/does-not-exist").status_code == 404)
    check("path traversal job id not 5xx / not 200", c.get("/jobs/..%2F..%2F.env").status_code in (400, 404, 422))
    tj = post_text(hu_txt).json()["job_id"]
    check("gradcam for text job 404", c.get(f"/jobs/{tj}/gradcam").status_code == 404)
    check("correction with bad label 422", c.post(f"/jobs/{tj}/correction", json={"label": "maybe"}).status_code == 422)
    check("correction good label 200", c.post(f"/jobs/{tj}/correction", json={"label": "authentic"}).status_code == 200)
    if key:
        r = httpx.post(args.base + "/analyze/text", json={"text": hu_txt})
        check("missing API key rejected", r.status_code in (401, 403), str(r.status_code))
        r = httpx.post(args.base + "/analyze/text", json={"text": hu_txt}, headers={"X-API-Key": "wrong"})
        check("wrong API key rejected", r.status_code in (401, 403), str(r.status_code))
    c.retry = False
    codes = [post_text("short").status_code for _ in range(45)]
    check("rate limit kicks in (429 within 45 rapid calls)", 429 in codes, f"codes seen: {sorted(set(codes))}")
    time.sleep(1)

    bad = [n for n, ok, _ in RESULTS if not ok]
    print(f"\n{len(RESULTS) - len(bad)}/{len(RESULTS)} passed")
    if bad:
        print("FAILED:", *bad, sep="\n  - ")
        sys.exit(1)


if __name__ == "__main__":
    main()
