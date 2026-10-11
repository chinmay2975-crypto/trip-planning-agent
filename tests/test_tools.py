import json

import httpx

from trip_planner.config import get_settings
from trip_planner.tools import currency, flights, search, weather
from trip_planner.tools.flights import FlightQuery


def mock_client(monkeypatch, module, handler, calls=None):
    def factory():
        def wrapped(request):
            if calls is not None:
                calls.append(request)
            return handler(request)
        return httpx.Client(transport=httpx.MockTransport(wrapped))
    monkeypatch.setattr(module, "http_client", factory)


def set_keys(monkeypatch, **keys):
    for k, v in keys.items():
        monkeypatch.setenv(k, v)
    get_settings.cache_clear()


def test_search_without_keys_returns_error():
    assert "error" in search.web_search("lisbon")


def test_tavily_search_parses_and_caches(monkeypatch):
    set_keys(monkeypatch, TAVILY_API_KEY="k")
    calls = []
    mock_client(monkeypatch, search, lambda r: httpx.Response(
        200, json={"results": [{"title": "T", "url": "u", "content": "c" * 1000}]}), calls)
    first = search.web_search("lisbon hotels")
    second = search.web_search("lisbon hotels")
    assert first == second == [{"title": "T", "url": "u", "snippet": "c" * 300}]
    assert len(calls) == 1  # second call served from the SQLite cache


def test_http_errors_become_payload_and_are_not_cached(monkeypatch):
    set_keys(monkeypatch, TAVILY_API_KEY="k")
    calls = []
    mock_client(monkeypatch, search, lambda r: httpx.Response(429, text="rate limited"), calls)
    assert "429" in search.web_search("q")["error"]
    search.web_search("q")
    assert len(calls) == 2


def test_weather_collapses_slots_to_daily(monkeypatch):
    set_keys(monkeypatch, OPENWEATHER_API_KEY="k")

    def slot(t, temp, cond):
        return {"dt_txt": t, "main": {"temp": temp}, "weather": [{"main": cond}]}

    def handler(r):
        if "geo" in r.url.path:
            return httpx.Response(200, json=[{"name": "Lisbon", "country": "PT", "lat": 1, "lon": 2}])
        return httpx.Response(200, json={"list": [
            slot("2026-11-01 09:00:00", 14, "Rain"), slot("2026-11-01 15:00:00", 19, "Rain"),
            slot("2026-11-01 21:00:00", 12, "Clouds"), slot("2026-11-02 12:00:00", 20, "Clear")]})

    mock_client(monkeypatch, weather, handler)
    out = json.loads(weather.WeatherTool()._run("Lisbon"))
    assert out["daily"][0] == {"date": "2026-11-01", "min_c": 12, "max_c": 19, "condition": "Rain"}
    assert len(out["daily"]) == 2


def test_currency_keyless_endpoint(monkeypatch):
    mock_client(monkeypatch, currency, lambda r: httpx.Response(
        200, json={"result": "success", "rates": {"USD": 1.1}}))
    assert currency.convert(100, "eur", "usd") == {"amount": 110.0, "currency": "USD", "rate": 1.1}
    assert currency.convert(5, "USD", "USD")["amount"] == 5


def test_flights_fall_back_to_estimate_when_amadeus_fails(monkeypatch):
    set_keys(monkeypatch, AMADEUS_CLIENT_ID="id", AMADEUS_CLIENT_SECRET="secret")
    monkeypatch.setattr(flights, "_amadeus_token", None)
    mock_client(monkeypatch, flights, lambda r: httpx.Response(401, json={"error": "invalid_client"}))
    monkeypatch.setattr(flights, "web_search",
                        lambda q, n: [{"title": "BOM-LIS from $540", "url": "u", "snippet": ""}])
    res = flights.search_flights(FlightQuery(origin="BOM", destination="LIS",
                                             departure_date="2026-11-01", return_date="2026-11-05"))
    assert res["provider"] == "estimate"
    assert res["fallback_reason"][0].startswith("amadeus: HTTP 401")


def test_amadeus_offers_are_normalised(monkeypatch):
    set_keys(monkeypatch, AMADEUS_CLIENT_ID="id", AMADEUS_CLIENT_SECRET="secret")
    monkeypatch.setattr(flights, "_amadeus_token", None)

    def seg(a, b, t):
        return {"departure": {"iataCode": a, "at": t}, "arrival": {"iataCode": b}}

    offer = {
        "validatingAirlineCodes": ["TP"],
        "itineraries": [
            {"segments": [seg("BOM", "DXB", "2026-11-01T02:00"), seg("DXB", "LIS", "2026-11-01T09:00")]},
            {"segments": [seg("LIS", "BOM", "2026-11-05T10:00")]},
        ],
        "price": {"grandTotal": "1200.00", "currency": "USD"},
    }

    def handler(r):
        if "oauth2" in r.url.path:
            return httpx.Response(200, json={"access_token": "t", "expires_in": 1799})
        assert r.url.params["maxPrice"] == "1400"
        return httpx.Response(200, json={"data": [offer]})

    mock_client(monkeypatch, flights, handler)
    res = flights.search_flights(FlightQuery(origin="bom", destination="lis", departure_date="2026-11-01",
                                             return_date="2026-11-05", adults=2, max_price_per_person=700))
    o = res["offers"][0]
    assert res["provider"] == "amadeus"
    assert (o["route"], o["stops"], o["price_per_person"]) == ("BOM-DXB-LIS-BOM", 1, 600.0)
