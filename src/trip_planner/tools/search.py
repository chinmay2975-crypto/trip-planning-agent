"""Web search via Tavily (preferred) or Serper, whichever key is configured."""

from __future__ import annotations

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..cache import cached
from ..config import get_settings
from ._common import error, http_client, safe_call, to_json


@cached("search")
def _tavily(query: str, max_results: int) -> list[dict] | dict:
    with http_client() as c:
        r = c.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {get_settings().tavily_api_key}"},
            json={"query": query, "max_results": max_results, "search_depth": "basic"},
        )
        r.raise_for_status()
    return [
        {"title": i.get("title"), "url": i.get("url"), "snippet": (i.get("content") or "")[:300]}
        for i in r.json().get("results", [])
    ]


@cached("search")
def _serper(query: str, max_results: int) -> list[dict] | dict:
    with http_client() as c:
        r = c.post(
            "https://google.serper.dev/search",
            headers={"X-API-KEY": get_settings().serper_api_key or ""},
            json={"q": query, "num": max_results},
        )
        r.raise_for_status()
    return [
        {"title": i.get("title"), "url": i.get("link"), "snippet": (i.get("snippet") or "")[:300]}
        for i in r.json().get("organic", [])[:max_results]
    ]


def web_search(query: str, max_results: int = 5) -> list[dict] | dict:
    s = get_settings()
    if s.tavily_api_key:
        return safe_call(_tavily, query, max_results)
    if s.serper_api_key:
        return safe_call(_serper, query, max_results)
    return error("No search API key configured (TAVILY_API_KEY or SERPER_API_KEY)")


# Tool arg schemas carry no numeric/length bounds: Groq validates tool calls
# against the schema server-side and rejects the whole call (400) on any
# violation, e.g. max_results=10 vs a maximum of 8. Limits live in code.
class SearchInput(BaseModel):
    query: str = Field(..., description="Search query, be specific (city, dates, budget)")


class WebSearchTool(BaseTool):
    name: str = "web_search"
    description: str = (
        "Search the web. Returns a JSON list of {title, url, snippet}. Use for visa rules, "
        "events, hotels, neighbourhoods, prices and anything not covered by other tools."
    )
    args_schema: type[BaseModel] = SearchInput

    def _run(self, query: str, **_: object) -> str:
        return to_json(web_search(query, 4))
