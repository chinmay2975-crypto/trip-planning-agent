import json

import httpx
import openai
import pytest

from trip_planner import llm as groq_llm
from trip_planner.llm import GroqCompletion, recover_answer

TOOLS = [{"type": "function", "function": {"name": "web_search", "parameters": {}}}]


def tool_use_failed(message: str, generation: dict | str) -> openai.BadRequestError:
    gen = generation if isinstance(generation, str) else json.dumps(generation)
    body = {"error": {"message": message, "code": "tool_use_failed", "failed_generation": gen}}
    response = httpx.Response(400, request=httpx.Request("POST", "https://api.groq.com"), json=body)
    return openai.BadRequestError(message, response=response, body=body)


def test_recover_answer_from_pseudo_json_tool():
    err = tool_use_failed("x", {"name": "json", "arguments": {"summary": "Lisbon"}}).body["error"]
    assert json.loads(recover_answer(err, TOOLS)) == {"summary": "Lisbon"}


def test_real_tool_calls_are_not_treated_as_answers():
    err = tool_use_failed("x", {"name": "web_search", "arguments": {"query": "q"}}).body["error"]
    assert recover_answer(err, TOOLS) is None
    assert recover_answer({"failed_generation": "not json"}, TOOLS) is None


@pytest.fixture
def completion(monkeypatch):
    llm = GroqCompletion(model="openai/gpt-oss-120b", provider="openai", api_key="k",
                         base_url="https://api.groq.com/openai/v1", custom_openai=True)
    calls = []

    def install(*outcomes):
        def fake_call(self, messages, tools=None, *a, **kw):
            outcome = outcomes[len(calls)]
            calls.append(tools)
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        monkeypatch.setattr(groq_llm.OpenAICompletion, "call", fake_call)
        return calls

    return llm, install


def test_pseudo_tool_answer_is_returned_without_retry(completion):
    llm, install = completion
    calls = install(tool_use_failed("not in request.tools", {"name": "JSON", "arguments": {"a": 1}}))
    assert json.loads(llm.call("hi", TOOLS)) == {"a": 1}
    assert len(calls) == 1


def test_other_tool_use_failures_are_retried(completion):
    llm, install = completion
    bad = tool_use_failed("Tool choice is none, but model called a tool",
                          {"name": "web_search", "arguments": {"query": "q"}})
    calls = install(bad, "final answer")
    assert llm.call("hi", TOOLS) == "final answer"
    assert len(calls) == 2


def test_gives_up_after_max_retries(completion):
    llm, install = completion
    bad = tool_use_failed("Failed to parse tool call arguments as JSON", '{"name": "web_search", "arguments": {')
    calls = install(bad, bad, bad)
    with pytest.raises(openai.BadRequestError):
        llm.call("hi", TOOLS)
    assert len(calls) == 3


def test_unrelated_errors_are_not_retried(completion):
    llm, install = completion
    body = {"error": {"message": "model not found", "code": "model_not_found"}}
    response = httpx.Response(400, request=httpx.Request("POST", "https://x"), json=body)
    calls = install(openai.BadRequestError("nf", response=response, body=body))
    with pytest.raises(openai.BadRequestError):
        llm.call("hi")
    assert len(calls) == 1


def rate_limited(message: str) -> openai.RateLimitError:
    response = httpx.Response(429, request=httpx.Request("POST", "https://api.groq.com"))
    return openai.RateLimitError(message, response=response, body=None)


TPD = ("Rate limit reached for model `openai/gpt-oss-20b` on tokens per day (TPD): "
       "Limit 200000, Used 198476. Please try again in 12m10.08s.")
TPM = "Rate limit reached on tokens per minute (TPM): Limit 8000. Please try again in 4.6s."


@pytest.fixture(autouse=True)
def clear_exhausted():
    groq_llm._exhausted.clear()
    yield
    groq_llm._exhausted.clear()


def test_parse_retry_seconds():
    assert groq_llm.parse_retry_seconds("Please try again in 12m10.08s.") == pytest.approx(730.08)
    assert groq_llm.parse_retry_seconds("try again in 1h2m3s") == 3723
    assert groq_llm.parse_retry_seconds("no hint") == groq_llm.DEFAULT_COOLDOWN_SECONDS


@pytest.fixture
def chained(monkeypatch):
    llm = GroqCompletion(model="openai/gpt-oss-20b", provider="openai", api_key="k",
                         base_url="https://api.groq.com/openai/v1", custom_openai=True,
                         fallback_models=["openai/gpt-oss-120b", "openai/gpt-oss-20b", "qwen/qwen3.8-27b"])
    models_called = []

    def install(outcomes: dict):
        def fake_call(self, messages, tools=None, *a, **kw):
            models_called.append(self.model)
            outcome = outcomes[self.model]
            if isinstance(outcome, Exception):
                raise outcome
            return outcome
        monkeypatch.setattr(groq_llm.OpenAICompletion, "call", fake_call)
        return models_called

    return llm, install


def test_daily_limit_switches_to_next_model(chained):
    llm, install = chained
    called = install({"openai/gpt-oss-20b": rate_limited(TPD), "openai/gpt-oss-120b": "from 120b"})
    assert llm.call("hi") == "from 120b"
    assert called == ["openai/gpt-oss-20b", "openai/gpt-oss-120b"]
    # Exhaustion is process-wide: a fresh instance skips 20b straight away.
    other = GroqCompletion(model="openai/gpt-oss-20b", provider="openai", api_key="k",
                           base_url="https://x", custom_openai=True,
                           fallback_models=["openai/gpt-oss-120b"])
    assert other.call("hi") == "from 120b"
    assert called[-1] == "openai/gpt-oss-120b"


def test_primary_is_used_again_once_quota_reopens(chained):
    llm, install = chained
    groq_llm.mark_exhausted("openai/gpt-oss-20b", -1)  # window already passed
    called = install({"openai/gpt-oss-20b": "from 20b"})
    assert llm.call("hi") == "from 20b" and called == ["openai/gpt-oss-20b"]


def test_all_models_exhausted_raises_clear_error(chained):
    llm, install = chained
    install({m: rate_limited(TPD) for m in ["openai/gpt-oss-20b", "openai/gpt-oss-120b", "qwen/qwen3.8-27b"]})
    with pytest.raises(groq_llm.QuotaExhaustedError, match="all models"):
        llm.call("hi")


def test_per_minute_limit_is_left_to_crewai_retry(chained):
    llm, install = chained
    called = install({"openai/gpt-oss-20b": rate_limited(TPM)})
    with pytest.raises(openai.RateLimitError):
        llm.call("hi")
    assert called == ["openai/gpt-oss-20b"] * len(called)
    assert not groq_llm.is_exhausted("openai/gpt-oss-20b")
