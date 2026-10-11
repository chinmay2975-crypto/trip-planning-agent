"""OpenRouteService (free key, no billing card): geocode places + travel-time matrix."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..cache import cached
from ..config import get_settings
from ._common import error, http_client, safe_call, to_json

ORS = "https://api.openrouteservice.org"


@cached("ors_geocode", ttl_hours=24 * 30)
def _geocode(text: str) -> dict:
    with http_client() as c:
        r = c.get(
            f"{ORS}/geocode/search",
            params={"api_key": get_settings().ors_api_key, "text": text, "size": 1},
        )
        r.raise_for_status()
    feats = r.json().get("features", [])
    if not feats:
        return error(f"Place not found: {text}")
    lon, lat = feats[0]["geometry"]["coordinates"]
    return {"label": feats[0]["properties"].get("label", text), "lon": lon, "lat": lat}


@cached("ors_matrix", ttl_hours=24 * 7)
def _matrix(coords: list[list[float]], profile: str) -> dict:
    with http_client() as c:
        r = c.post(
            f"{ORS}/v2/matrix/{profile}",
            headers={"Authorization": get_settings().ors_api_key or ""},
            json={"locations": coords, "metrics": ["duration", "distance"]},
        )
        r.raise_for_status()
    return r.json()


def travel_matrix(places: list[str], city: str, profile: str) -> dict:
    located, missing = [], []
    for p in places:
        g = safe_call(_geocode, f"{p}, {city}")
        (missing if "error" in g else located).append(g if "error" not in g else p)
    if len(located) < 2:
        return error(f"Need at least 2 geocoded places; not found: {missing}")

    m = safe_call(_matrix, [[g["lon"], g["lat"]] for g in located], profile)
    if "error" in m:
        return m
    names = [g["label"].split(",")[0] for g in located]
    # Consecutive legs are what an itinerary needs; full matrix helps reordering.
    legs = [
        {
            "from": names[i],
            "to": names[i + 1],
            "minutes": round((m["durations"][i][i + 1] or 0) / 60),
            "km": round((m["distances"][i][i + 1] or 0) / 1000, 1),
        }
        for i in range(len(names) - 1)
    ]
    minutes = [[round((d or 0) / 60) for d in row] for row in m["durations"]]
    return {"profile": profile, "places": names, "legs": legs, "minutes_matrix": minutes, "not_found": missing}


class RoutingInput(BaseModel):
    # No length bounds in the schema (Groq rejects violating calls); enforced in _run.
    places: list[str] = Field(..., description="2 to 10 place names in visiting order")
    city: str = Field(..., description="City the places are in, for geocoding")
    profile: Literal["foot-walking", "driving-car", "cycling-regular"] = "foot-walking"


class RoutingTool(BaseTool):
    name: str = "travel_times"
    description: str = (
        "Get travel times between places in a city. Returns consecutive legs "
        "(minutes, km) and a full minutes matrix to help reorder stops and group "
        "nearby sights on the same day."
    )
    args_schema: type[BaseModel] = RoutingInput

    def _run(self, places: list[str], city: str, profile: str = "foot-walking") -> str:
        if not get_settings().ors_api_key:
            return to_json(error("ORS_API_KEY not set; estimate travel times from geography"))
        if len(places) < 2:
            return to_json(error("Give at least 2 places"))
        return to_json(travel_matrix(places[:10], city, profile))
