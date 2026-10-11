"""Flight search behind a provider interface.

Free flight data is the project's known weak spot: Amadeus Self-Service (test
environment, cached/limited inventory) is used when credentials exist and
responds; otherwise the search-based provider returns web snippets with fares
for the agent to estimate from. Results are labelled with their source so the
UI can show which numbers are real quotes and which are estimates.
"""

from __future__ import annotations

import time
from typing import Protocol

from pydantic import BaseModel, Field, field_validator
from crewai.tools import BaseTool

from ..cache import cached
from ..config import get_settings
from ._common import error, http_client, safe_call, to_json
from .search import web_search


class FlightQuery(BaseModel):
    origin: str = Field(..., description="Origin IATA airport/city code, e.g. 'BOM'")
    destination: str = Field(..., description="Destination IATA code, e.g. 'LIS'")
    departure_date: str = Field(..., description="YYYY-MM-DD")
    return_date: str = Field(..., description="YYYY-MM-DD")
    adults: int = Field(1, description="Number of adult travellers")
    currency: str = Field("USD", description="ISO currency for prices")
    # Plain number, not Optional: CrewAI sends tool schemas in strict mode (every
    # field required), and models tend to omit a nullable field instead of
    # sending null, which Groq rejects with a 400.
    # No ge/le bounds either: Groq rejects calls that break schema bounds, so
    # values are clamped below instead.
    max_price_per_person: float = Field(0, description="Price cap per person; 0 means no cap")

    @field_validator("adults")
    @classmethod
    def _clamp_adults(cls, v: int) -> int:
        return min(max(v, 1), 9)

    @field_validator("max_price_per_person")
    @classmethod
    def _clamp_cap(cls, v: float) -> float:
        return max(v, 0.0)


class FlightProvider(Protocol):
    name: str

    def search(self, q: FlightQuery) -> dict: ...


_amadeus_token: tuple[str, float] | None = None


def _amadeus_auth() -> str:
    global _amadeus_token
    if _amadeus_token and _amadeus_token[1] > time.time():
        return _amadeus_token[0]
    s = get_settings()
    with http_client() as c:
        r = c.post(
            f"{s.amadeus_base_url}/v1/security/oauth2/token",
            data={
                "grant_type": "client_credentials",
                "client_id": s.amadeus_client_id,
                "client_secret": s.amadeus_client_secret,
            },
        )
        r.raise_for_status()
    body = r.json()
    _amadeus_token = (body["access_token"], time.time() + body["expires_in"] - 60)
    return body["access_token"]


class AmadeusProvider:
    name = "amadeus"

    def search(self, q: FlightQuery) -> dict:
        return safe_call(_amadeus_search, q.model_dump_json())


@cached("amadeus_flights", ttl_hours=6)
def _amadeus_search(query_json: str) -> dict:
    q = FlightQuery.model_validate_json(query_json)
    params = {
        "originLocationCode": q.origin.upper(),
        "destinationLocationCode": q.destination.upper(),
        "departureDate": q.departure_date,
        "returnDate": q.return_date,
        "adults": q.adults,
        "currencyCode": q.currency.upper(),
        "max": 5,
    }
    if q.max_price_per_person:
        # Amadeus maxPrice applies to the whole offer (all travellers).
        params["maxPrice"] = int(q.max_price_per_person * q.adults)
    with http_client() as c:
        r = c.get(
            f"{get_settings().amadeus_base_url}/v2/shopping/flight-offers",
            params=params,
            headers={"Authorization": f"Bearer {_amadeus_auth()}"},
        )
        r.raise_for_status()

    offers = []
    for o in r.json().get("data", []):
        legs = o["itineraries"]
        segs = [s for leg in legs for s in leg["segments"]]
        offers.append(
            {
                "airline": ",".join(sorted(set(o.get("validatingAirlineCodes", [])))),
                "route": "-".join([segs[0]["departure"]["iataCode"]] + [s["arrival"]["iataCode"] for s in segs]),
                "outbound": legs[0]["segments"][0]["departure"]["at"],
                "inbound": legs[-1]["segments"][0]["departure"]["at"] if len(legs) > 1 else "",
                "stops": max(len(leg["segments"]) - 1 for leg in legs),
                "price_per_person": round(float(o["price"]["grandTotal"]) / q.adults, 2),
                "currency": o["price"]["currency"],
                "source": "amadeus",
            }
        )
    return {"provider": "amadeus", "offers": offers}


class SearchEstimateProvider:
    """Fallback: web results about fares on this route for the agent to estimate from."""

    name = "estimate"

    def search(self, q: FlightQuery) -> dict:
        month = q.departure_date[:7]
        cap = f" under {q.max_price_per_person:.0f} {q.currency}" if q.max_price_per_person else ""
        results = web_search(
            f"cheap round trip flights {q.origin} to {q.destination} {month} price{cap}", 4
        )
        if isinstance(results, dict):
            return results
        return {
            "provider": "estimate",
            "note": (
                "No live quotes exist; this is all the data available, so do not repeat "
                "this search. Estimate realistic round-trip prices per person from these "
                "snippets and set source='estimate: <url>'. Do not invent flight numbers "
                "or times: put only the dates in outbound/inbound."
            ),
            "results": results,
        }


def get_providers() -> list[FlightProvider]:
    s = get_settings()
    providers: list[FlightProvider] = []
    if s.amadeus_client_id and s.amadeus_client_secret:
        providers.append(AmadeusProvider())
    providers.append(SearchEstimateProvider())
    return providers


def search_flights(q: FlightQuery) -> dict:
    """Try providers in order; fall through on errors or empty results."""
    errors = []
    for p in get_providers():
        res = p.search(q)
        if "error" in res:
            errors.append(f"{p.name}: {res['error']}")
            continue
        if res.get("provider") == "amadeus" and not res.get("offers"):
            errors.append("amadeus: no offers")
            continue
        if errors:
            res["fallback_reason"] = errors
        return res
    return error("; ".join(errors) or "no flight providers available")


class FlightSearchTool(BaseTool):
    name: str = "search_flights"
    description: str = (
        "Search round-trip flights between two IATA codes. Returns live offers "
        "(provider='amadeus') or web snippets to estimate from (provider='estimate'). "
        "Call it once per route; set max_price_per_person (0 = no cap) only when "
        "re-searching cheaper options."
    )
    args_schema: type[BaseModel] = FlightQuery

    def _run(self, **kwargs) -> str:
        return to_json(search_flights(FlightQuery(**kwargs)))
