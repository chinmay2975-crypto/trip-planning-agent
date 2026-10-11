"""Tiny SQLite cache for external API responses (saves free-tier quota in dev)."""

from __future__ import annotations

import functools
import hashlib
import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Callable, Iterator

from .config import get_settings

_lock = threading.Lock()


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    path = get_settings().cache_path
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS cache (key TEXT PRIMARY KEY, value TEXT, expires REAL)"
        )
        yield conn
        conn.commit()
    finally:
        conn.close()


def _key(namespace: str, args: tuple, kwargs: dict) -> str:
    raw = json.dumps([namespace, args, kwargs], sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def cached(namespace: str, ttl_hours: float | None = None) -> Callable:
    """Cache a function's JSON-serialisable return value by its arguments.

    Results containing an ``"error"`` key are not cached, so transient
    failures are retried on the next call.
    """

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            ttl = (ttl_hours or get_settings().cache_ttl_hours) * 3600
            key = _key(namespace, args, kwargs)
            with _lock, _connect() as conn:
                row = conn.execute(
                    "SELECT value, expires FROM cache WHERE key = ?", (key,)
                ).fetchone()
            if row and row[1] > time.time():
                return json.loads(row[0])

            result = fn(*args, **kwargs)
            if not (isinstance(result, dict) and "error" in result):
                with _lock, _connect() as conn:
                    conn.execute(
                        "INSERT OR REPLACE INTO cache VALUES (?, ?, ?)",
                        (key, json.dumps(result, default=str), time.time() + ttl),
                    )
            return result

        return wrapper

    return decorator
