"""Build / update the knowledge base (RAG layers 1 and 2).

Usage:
    python rag/ingest.py --kb-dir rag/knowledge_base          # ingest every .md/.txt file
    python rag/ingest.py --kb-dir rag/knowledge_base --reset  # drop the collection first
    python rag/ingest.py --file path/to/note.md

Each note may start with a front-matter block:
    ---
    modality: text | image | general
    ---
The evidence type comes from the parent folder: fingerprints/ -> generator_fingerprint,
literature/ -> literature (anything else falls back to the file name).
"""
from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from rag.store import KB_COLLECTION, get_client, get_kb  # noqa: E402

_FRONT_MATTER = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)


def parse_front_matter(text: str) -> tuple[dict, str]:
    match = _FRONT_MATTER.match(text)
    if not match:
        return {}, text
    meta = {}
    for line in match.group(1).splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            meta[key.strip().lower()] = value.strip()
    return meta, text[match.end():]


def chunk_text(text: str, chunk_words: int = 220, overlap_words: int = 40) -> list[str]:
    """Split on markdown headings first, then into overlapping word windows."""
    sections = [s.strip() for s in re.split(r"\n(?=#{1,3} )", text) if s.strip()]
    chunks: list[str] = []
    for section in sections:
        words = section.split()
        if len(words) <= chunk_words:
            chunks.append(section)
            continue
        step = max(chunk_words - overlap_words, 1)
        for start in range(0, len(words), step):
            piece = " ".join(words[start:start + chunk_words])
            chunks.append(piece)
            if start + chunk_words >= len(words):
                break
    return chunks


def evidence_type(path: Path) -> str:
    parts = {p.lower() for p in path.parts}
    name = path.stem.lower()
    if "fingerprints" in parts or "fingerprint" in name or "generator" in name:
        return "generator_fingerprint"
    if "cases" in parts or "case" in name:
        return "case_study"
    return "literature"


def ingest_file(path: Path, collection=None) -> int:
    collection = collection or get_kb()
    raw = path.read_text(encoding="utf-8", errors="ignore")
    meta, body = parse_front_matter(raw)
    modality = meta.get("modality", "general")
    if modality not in {"text", "image", "general"}:
        raise ValueError(f"{path}: modality must be text, image or general (got '{modality}')")
    chunks = chunk_text(body)
    if not chunks:
        return 0
    collection.delete(where={"source": path.stem})          # drop stale chunks of this file
    kind = evidence_type(path)
    ids = [hashlib.sha256(f"{path.stem}:{i}".encode()).hexdigest()[:16] for i in range(len(chunks))]
    metas = [{"source": path.stem, "chunk": i, "evidence_type": kind, "modality": modality} for i in range(len(chunks))]
    collection.upsert(ids=ids, documents=chunks, metadatas=metas)
    return len(chunks)


def ingest_dir(kb_dir: Path, collection=None) -> int:
    total = 0
    for f in sorted(kb_dir.rglob("*")):
        if f.suffix.lower() in {".md", ".txt"} and f.name.lower() != "readme.md":
            n = ingest_file(f, collection)
            print(f"  ingested {f.relative_to(kb_dir)}: {n} chunks")
            total += n
    return total


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--kb-dir", type=Path)
    ap.add_argument("--file", type=Path)
    ap.add_argument("--reset", action="store_true", help="delete the knowledge-base collection first")
    args = ap.parse_args()
    if not args.kb_dir and not args.file:
        ap.error("give --kb-dir or --file")
    if args.reset:
        try:
            get_client().delete_collection(KB_COLLECTION)
            print("Deleted existing knowledge-base collection.")
        except Exception:
            pass
    total = ingest_file(args.file) if args.file else ingest_dir(args.kb_dir)
    print(f"Total chunks upserted: {total}; collection size: {get_kb().count()}")


if __name__ == "__main__":
    main()
