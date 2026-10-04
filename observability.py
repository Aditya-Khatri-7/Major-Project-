"""Structured JSON logging: one line per event, written to stderr."""
from __future__ import annotations

import json
import logging
import sys
import time
from contextlib import contextmanager

from config import settings


class _JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "ts": self.formatTime(record, "%Y-%m-%dT%H:%M:%S"),
            "level": record.levelname,
            "logger": record.name,
            "event": record.getMessage(),
        }
        fields = getattr(record, "fields", None)
        if fields:
            payload.update(fields)
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


_configured = False


def _configure() -> None:
    global _configured
    if _configured:
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(_JsonFormatter())
    root = logging.getLogger("forensics")
    root.setLevel(settings.log_level.upper())
    root.addHandler(handler)
    root.propagate = False
    _configured = True


def get_logger(name: str) -> logging.Logger:
    _configure()
    return logging.getLogger(f"forensics.{name}")


def log_event(logger: logging.Logger, event: str, level: int = logging.INFO, **fields) -> None:
    logger.log(level, event, extra={"fields": fields})


@contextmanager
def timed(logger: logging.Logger, event: str, **fields):
    """Log `event` with a latency_ms field when the block exits."""
    start = time.perf_counter()
    try:
        yield
    finally:
        log_event(logger, event, latency_ms=round((time.perf_counter() - start) * 1000, 1), **fields)
