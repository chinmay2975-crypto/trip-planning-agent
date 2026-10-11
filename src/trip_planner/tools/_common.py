"""Shared helpers for tools: one httpx client config and compact JSON output."""

from __future__ import annotations

import json
from typing import Any

import httpx

from ..config import get_settings


def http_client() -> httpx.Client:
    return httpx.Client(timeout=get_settings().http_timeout, follow_redirects=True)


def to_json(data: Any) -> str:
    """Compact JSON for tool output; small models cope better with short context."""
    return json.dumps(data, separators=(",", ":"), ensure_ascii=False, default=str)


def error(msg: str) -> dict[str, str]:
    return {"error": msg}


def safe_call(fn, *args, **kwargs) -> dict | list:
    """Run an API call, turning HTTP/network failures into an error payload.

    Tools must not raise: an exception aborts the agent's step, while an error
    payload lets the agent adapt (retry, change query, or estimate).
    """
    try:
        return fn(*args, **kwargs)
    except httpx.HTTPStatusError as e:
        return error(f"HTTP {e.response.status_code}: {e.response.text[:200]}")
    except httpx.HTTPError as e:
        return error(f"Network error: {e}")
