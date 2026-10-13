"""Groq-specific LLM wrapper.

Groq validates tool calls server-side and fails the whole request with
``400 tool_use_failed`` when a model's tool call is invalid. The gpt-oss models
hit this regularly inside CrewAI:

* they "call" a non-existent tool such as ``json`` to deliver their final
  structured answer, so the answer is in the rejected call's arguments;
* they call a real tool on the forced-final-answer turn, which is sent with
  ``tool_choice="none"``;
* they occasionally emit malformed JSON arguments.

The first case is recovered from Groq's ``failed_generation``. The other two are
sampling glitches that a retry usually fixes. Without this, one bad generation
aborts the whole agent step and the hierarchical manager re-delegates, which
burns free-tier quota.

Groq's free tier also caps tokens *per day, per model*. When a model's daily
quota runs out, the call switches to the next model in ``fallback_models``.
Exhaustion is tracked process-wide, so every agent skips that model until its
quota window reopens. Per-minute limits are left to CrewAI's own retry.
"""

from __future__ import annotations

import json
import logging
import re
import threading
import time
from typing import Any

import openai
from crewai.llms.providers.openai.completion import OpenAICompletion
from pydantic import PrivateAttr

log = logging.getLogger(__name__)

MAX_RETRIES = 2
# Names gpt-oss invents for "here is my structured final answer".
ANSWER_PSEUDO_TOOLS = {"json", "final_answer", "answer", "response", "output", "result"}


DEFAULT_COOLDOWN_SECONDS = 3600.0
# Groq's daily window is rolling: "try again in 4m" only means room for that one
# request, and a nearly full quota is hit again right away. Bench the model for
# at least this long to avoid flip-flopping back to it.
MIN_COOLDOWN_SECONDS = 900.0

# model -> unix time its daily quota is expected to be usable again
_exhausted: dict[str, float] = {}
_exhausted_lock = threading.Lock()


class QuotaExhaustedError(RuntimeError):
    """Every model in the fallback chain is out of daily quota."""


def is_daily_limit(e: Exception) -> bool:
    msg = str(e).lower()
    return isinstance(e, openai.RateLimitError) and ("per day" in msg or "(tpd)" in msg or "(rpd)" in msg)


def parse_retry_seconds(message: str) -> float:
    """Seconds from Groq's 'Please try again in 1h2m13.5s' hint."""
    m = re.search(r"try again in\s+(?:(\d+)h)?(?:(\d+)m)?(?:([\d.]+)s)?", message)
    if not m or not any(m.groups()):
        return DEFAULT_COOLDOWN_SECONDS
    h, mins, secs = (float(g) if g else 0.0 for g in m.groups())
    return h * 3600 + mins * 60 + secs


def mark_exhausted(model: str, seconds: float) -> None:
    with _exhausted_lock:
        _exhausted[model] = time.time() + seconds


def is_exhausted(model: str) -> bool:
    with _exhausted_lock:
        until = _exhausted.get(model)
        if until and until <= time.time():
            del _exhausted[model]
            return False
        return until is not None


def _tool_use_error(e: Exception) -> dict | None:
    if not isinstance(e, openai.BadRequestError):
        return None
    body = e.body if isinstance(e.body, dict) else {}
    err = body.get("error", body)
    return err if isinstance(err, dict) and err.get("code") == "tool_use_failed" else None


def _tool_names(tools: list | None) -> set[str]:
    names = set()
    for t in tools or []:
        fn = t.get("function", t) if isinstance(t, dict) else {}
        if isinstance(fn, dict) and fn.get("name"):
            names.add(fn["name"])
    return names


def recover_answer(err: dict, tools: list | None) -> str | None:
    """Return the final answer if the model wrapped it in a pseudo-tool call."""
    try:
        gen = json.loads(err.get("failed_generation") or "")
    except (TypeError, ValueError):
        return None
    if not isinstance(gen, dict):
        return None
    name = str(gen.get("name", ""))
    if name.lower() not in ANSWER_PSEUDO_TOOLS or name in _tool_names(tools):
        return None
    args = gen.get("arguments")
    return args if isinstance(args, str) else json.dumps(args, ensure_ascii=False)


class GroqCompletion(OpenAICompletion):
    """OpenAI-compatible completion that tolerates Groq's tool-call rejections
    and falls back to other models when one runs out of daily quota."""

    fallback_models: list[str] = []
    _primary: str | None = PrivateAttr(default=None)

    def call(self, messages: Any, tools: list | None = None, *args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            self._select_model()
            try:
                return super().call(messages, tools, *args, **kwargs)
            except openai.RateLimitError as e:
                self._on_rate_limit(e)  # re-raises unless we can switch model
            except openai.BadRequestError as e:
                answer = self._handle(e, tools, attempt)
                attempt += 1
                if answer is not None:
                    return answer

    async def acall(self, messages: Any, tools: list | None = None, *args: Any, **kwargs: Any) -> Any:
        attempt = 0
        while True:
            self._select_model()
            try:
                return await super().acall(messages, tools, *args, **kwargs)
            except openai.RateLimitError as e:
                self._on_rate_limit(e)
            except openai.BadRequestError as e:
                answer = self._handle(e, tools, attempt)
                attempt += 1
                if answer is not None:
                    return answer

    def _chain(self) -> list[str]:
        self._primary = self._primary or self.model
        return [self._primary] + [m for m in self.fallback_models if m != self._primary]

    def _select_model(self) -> None:
        """Use the first model in the chain that still has daily quota."""
        for model in self._chain():
            if not is_exhausted(model):
                if model != self.model:
                    log.warning("Switching %s -> %s (daily quota exhausted)", self.model, model)
                    self.model = model
                return
        raise QuotaExhaustedError(
            f"Daily token quota exhausted for all models: {', '.join(self._chain())}. "
            "Try again later or set LLM_MODEL to another provider."
        )

    def _on_rate_limit(self, e: openai.RateLimitError) -> None:
        if not is_daily_limit(e) or len(self._chain()) == 1:
            raise e
        mark_exhausted(self.model, max(parse_retry_seconds(str(e)), MIN_COOLDOWN_SECONDS))

    def _handle(self, e: openai.BadRequestError, tools: list | None, attempt: int) -> str | None:
        """Recovered answer, None to retry, or re-raise when out of options."""
        err = _tool_use_error(e)
        if err is None:
            raise e
        answer = recover_answer(err, tools)
        if answer is not None:
            log.warning("Recovered final answer from pseudo tool call (%s)", self.model)
            return answer
        if attempt >= MAX_RETRIES:
            raise e
        log.warning("Groq tool_use_failed (%s), retrying: %s", self.model, err.get("message", "")[:120])
        return None
