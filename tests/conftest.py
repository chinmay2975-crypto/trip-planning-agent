from datetime import date

import pytest

from trip_planner.config import get_settings
from trip_planner.models import TripRequest


@pytest.fixture(autouse=True)
def isolated_settings(tmp_path, monkeypatch):
    """Fresh settings per test: temp cache DB and no real API keys."""
    for key in ["TAVILY_API_KEY", "SERPER_API_KEY", "OPENWEATHER_API_KEY", "ORS_API_KEY",
                "EXCHANGERATE_API_KEY", "AMADEUS_CLIENT_ID", "AMADEUS_CLIENT_SECRET"]:
        monkeypatch.setenv(key, "")
    monkeypatch.setenv("CACHE_PATH", str(tmp_path / "cache.sqlite"))
    monkeypatch.setenv("MAX_BUDGET_ITERATIONS", "3")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


@pytest.fixture
def request_1000() -> TripRequest:
    # 4 nights / 5 days, 1 traveller
    return TripRequest(origin="Mumbai", destination="Lisbon", start_date=date(2026, 11, 1),
                       end_date=date(2026, 11, 5), budget=1000)
