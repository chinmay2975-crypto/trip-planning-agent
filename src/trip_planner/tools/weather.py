"""OpenWeatherMap free tier: geocoding + 5-day / 3-hour forecast."""

from __future__ import annotations

from collections import defaultdict

from pydantic import BaseModel, Field
from crewai.tools import BaseTool

from ..cache import cached
from ..config import get_settings
from ._common import error, http_client, safe_call, to_json

OWM = "https://api.openweathermap.org"


@cached("weather", ttl_hours=3)
def _forecast(city: str) -> dict:
    key = get_settings().openweather_api_key
    with http_client() as c:
        g = c.get(f"{OWM}/geo/1.0/direct", params={"q": city, "limit": 1, "appid": key})
        g.raise_for_status()
        places = g.json()
        if not places:
            return error(f"City not found: {city}")
        lat, lon = places[0]["lat"], places[0]["lon"]
        f = c.get(
            f"{OWM}/data/2.5/forecast",
            params={"lat": lat, "lon": lon, "units": "metric", "appid": key},
        )
        f.raise_for_status()

    # Collapse 3-hour slots into daily min/max + most common condition.
    days: dict[str, dict] = defaultdict(lambda: {"temps": [], "conditions": []})
    for slot in f.json().get("list", []):
        day = slot["dt_txt"][:10]
        days[day]["temps"].append(slot["main"]["temp"])
        days[day]["conditions"].append(slot["weather"][0]["main"])
    daily = [
        {
            "date": d,
            "min_c": round(min(v["temps"]), 1),
            "max_c": round(max(v["temps"]), 1),
            "condition": max(set(v["conditions"]), key=v["conditions"].count),
        }
        for d, v in sorted(days.items())
    ]
    return {"city": places[0].get("name", city), "country": places[0].get("country"), "daily": daily}


class WeatherInput(BaseModel):
    city: str = Field(..., description="City name, optionally with country code, e.g. 'Lisbon,PT'")


class WeatherTool(BaseTool):
    name: str = "weather_forecast"
    description: str = (
        "Get the 5-day weather forecast (daily min/max °C and condition) for a city. "
        "Only covers the next 5 days; for trips further out, use web_search for typical "
        "climate in the travel month."
    )
    args_schema: type[BaseModel] = WeatherInput

    def _run(self, city: str) -> str:
        if not get_settings().openweather_api_key:
            return to_json(error("OPENWEATHER_API_KEY not set; use web_search for climate"))
        return to_json(safe_call(_forecast, city))
