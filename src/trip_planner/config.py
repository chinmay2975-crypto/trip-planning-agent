"""Settings and LLM factory.

Every external key is optional at import time so tests and the API can start
without a full .env; tools report a clear error when their key is missing.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[2]

GROQ_BASE_URL = "https://api.groq.com/openai/v1"


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env", env_file_encoding="utf-8", extra="ignore"
    )

    llm_model: str = "groq/openai/gpt-oss-120b"
    # Specialists can run on a smaller model. On Groq each model has its own
    # free daily token quota, so this roughly doubles the budget per run.
    specialist_llm_model: str | None = "groq/openai/gpt-oss-20b"
    crew_process: Literal["hierarchical", "sequential"] = "hierarchical"
    # Groq models to fall back to (in order) when a model's daily quota runs out.
    groq_fallback_models: str = "openai/gpt-oss-120b,openai/gpt-oss-20b,qwen/qwen3.8-27b"
    llm_temperature: float = 0.2
    groq_api_key: str | None = None
    gemini_api_key: str | None = None

    tavily_api_key: str | None = None
    serper_api_key: str | None = None
    openweather_api_key: str | None = None
    ors_api_key: str | None = None
    exchangerate_api_key: str | None = None
    amadeus_client_id: str | None = None
    amadeus_client_secret: str | None = None
    amadeus_base_url: str = "https://test.api.amadeus.com"

    max_budget_iterations: int = 3
    max_rpm: int = 20
    cache_ttl_hours: int = 24
    cache_path: Path = PROJECT_ROOT / ".cache" / "api_cache.sqlite"
    http_timeout: float = 20.0


@lru_cache
def get_settings() -> Settings:
    return Settings()


def get_llm(model: str | None = None):
    """Build an LLM from a ``provider/model`` string (default ``LLM_MODEL``).

    CrewAI 1.x has no native Groq provider, but Groq serves an
    OpenAI-compatible API, so ``groq/<model>`` is routed through the OpenAI
    provider with Groq's base URL (no LiteLLM dependency needed), wrapped in
    ``GroqCompletion`` to survive Groq's tool-call rejections.
    """
    from crewai import LLM

    from .llm import GroqCompletion

    s = get_settings()
    model = model or s.llm_model
    provider, _, name = model.partition("/")
    if provider == "groq":
        if not s.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is not set (see .env.example)")
        return GroqCompletion(
            model=name,
            fallback_models=[m.strip() for m in s.groq_fallback_models.split(",") if m.strip()],
            provider="openai",
            base_url=GROQ_BASE_URL,
            api_key=s.groq_api_key,
            custom_openai=True,
            temperature=s.llm_temperature,
        )
    if provider in {"gemini", "google"}:
        if not s.gemini_api_key:
            raise RuntimeError("GEMINI_API_KEY is not set (see .env.example)")
        return LLM(model=model, api_key=s.gemini_api_key, temperature=s.llm_temperature)
    return LLM(model=model, temperature=s.llm_temperature)


def get_specialist_llm():
    s = get_settings()
    return get_llm(s.specialist_llm_model or s.llm_model)
