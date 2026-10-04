"""API hardening: optional API-key auth, per-client rate limit, and retention of uploaded files.

All three are controlled from `config.Settings` and are off or generous by default, so local use is unchanged.
"""
from __future__ import annotations

import hmac
import logging
import shutil
import threading
import time
from collections import defaultdict, deque
from pathlib import Path

from fastapi import HTTPException, Request

from agents.common import log
from config import settings
from observability import log_event

_hits: dict[str, deque] = defaultdict(deque)
_hits_lock = threading.Lock()


def require_api_key(request: Request) -> None:
    """If API_KEY is configured, every protected endpoint needs a matching `X-API-Key` header."""
    expected = settings.api_key
    if not expected:
        return
    supplied = request.headers.get("x-api-key", "")
    if not hmac.compare_digest(supplied.encode(), expected.encode()):
        raise HTTPException(401, "Missing or invalid API key.")


def rate_limit(request: Request) -> None:
    """Sliding one-minute window per client address. RATE_LIMIT_PER_MIN=0 disables it."""
    limit = settings.rate_limit_per_min
    if limit <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _hits_lock:
        window = _hits[client]
        while window and now - window[0] > 60.0:
            window.popleft()
        if len(window) >= limit:
            raise HTTPException(429, "Too many requests. Slow down and retry in a minute.")
        window.append(now)


def reset_rate_limit() -> None:
    with _hits_lock:
        _hits.clear()


def purge_old_jobs(jobs_dir: str | Path | None = None, days: int | None = None) -> int:
    """Delete job folders (uploaded images, Grad-CAM) older than `days`. 0 disables retention cleanup."""
    days = settings.job_retention_days if days is None else days
    root = Path(jobs_dir or settings.jobs_dir)
    if days <= 0 or not root.is_dir():
        return 0
    cutoff = time.time() - days * 86400
    removed = 0
    for child in root.iterdir():
        try:
            if child.is_dir() and child.stat().st_mtime < cutoff:
                shutil.rmtree(child)
                removed += 1
        except OSError as exc:
            log_event(log, "job_purge_failed", level=logging.WARNING, path=str(child), error=str(exc))
    if removed:
        log_event(log, "jobs_purged", removed=removed, older_than_days=days)
    return removed
