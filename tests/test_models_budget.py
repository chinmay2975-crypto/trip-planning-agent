from datetime import date

import pytest
from pydantic import ValidationError

from trip_planner.budget import compute_budget, replan_caps
from trip_planner.models import Activity, DayPlan, FlightOption, Itinerary, StayOption, TripRequest


def test_trip_request_rejects_bad_dates():
    with pytest.raises(ValidationError):
        TripRequest(origin="A", destination="B", start_date=date(2026, 1, 5),
                    end_date=date(2026, 1, 5), budget=500)


def test_trip_request_nights_days_and_currency():
    r = TripRequest(origin="A", destination="B", start_date=date(2026, 1, 1),
                    end_date=date(2026, 1, 4), budget=500, currency="eur")
    assert (r.nights, r.days, r.currency) == (3, 4, "EUR")


def test_compute_budget_rolls_up_all_categories():
    req = TripRequest(origin="A", destination="B", start_date=date(2026, 1, 1),
                      end_date=date(2026, 1, 4), budget=1000, travelers=2)
    itin = Itinerary(
        days=[DayPlan(day=1, activities=[Activity(name="Museum", cost_per_person=15),
                                         Activity(name="Walk")])],
        food_per_person_per_day=30,
        local_transport_per_day=10,
    )
    r = compute_budget(req, FlightOption(airline="X", route="A-B", price_per_person=200),
                       StayOption(name="H", price_per_night=100), itin)
    # flights 400, stay 300, activities 30, food 30*4*2=240, transport 40
    assert (r.flights, r.accommodation, r.activities, r.food, r.local_transport) == (400, 300, 30, 240, 40)
    assert r.total == 1010 and r.over_budget and r.overage == 10


def test_compute_budget_handles_missing_parts(request_1000):
    r = compute_budget(request_1000, None, None, None)
    assert r.total == 0 and not r.over_budget


def test_replan_caps_split_overage_proportionally(request_1000):
    r = compute_budget(request_1000, FlightOption(airline="X", route="r", price_per_person=900),
                       StayOption(name="H", price_per_night=150), None)
    assert r.overage == 500  # 900 + 600 - 1000
    max_flight, max_stay = replan_caps(request_1000, r)
    # cut 525 split 60/40 across flights (900) and stay (600)
    assert max_flight == pytest.approx(585.0)
    assert max_stay == pytest.approx(97.5)  # (600 - 210) / 4 nights
    assert max_flight + max_stay * request_1000.nights < request_1000.budget


def test_flight_option_coerces_structured_times():
    f = FlightOption(airline="SK", route="BOM-LIS-BOM", price_per_person=287,
                     outbound={"date": "2026-11-15", "time": "02:30"}, inbound=None)
    assert (f.outbound, f.inbound) == ("2026-11-15 02:30", "")
