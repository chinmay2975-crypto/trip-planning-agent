from datetime import date
from pathlib import Path

from streamlit.testing.v1 import AppTest

from trip_planner.models import (Activity, BudgetReport, DayPlan, DestinationInfo, FlightOption,
                                 Itinerary, StayOption, TripPlan, TripRequest)

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def sample_plan() -> dict:
    req = TripRequest(origin="Mumbai", destination="Lisbon", start_date=date(2026, 11, 15),
                      end_date=date(2026, 11, 19), budget=800, preferences=["food"])
    flight = FlightOption(airline="TK", route="BOM-LIS-BOM", outbound="2026-11-15",
                          inbound="2026-11-19", stops=1, price_per_person=340, source="estimate: https://x")
    stay = StayOption(name="Alface Hostel", neighborhood="Baixa", price_per_night=35)
    first = BudgetReport(budget=800, flights=340, accommodation=600, total=940, over_budget=True, overage=140)
    final = BudgetReport(budget=800, flights=340, accommodation=140, food=200, total=680,
                         suggested_trims=["Skip the Oceanarium"])
    return TripPlan(
        request=req,
        destination=DestinationInfo(summary="Hilly, historic capital.", local_events=["LEFFEST"],
                                    neighborhoods=["Baixa", "Alfama"], tips=["Buy a Viva Viagem card"]),
        flights=[flight], stays=[stay], chosen_flight=flight, chosen_stay=stay,
        itinerary=Itinerary(days=[DayPlan(day=i, theme=f"Day {i}", activities=[
            Activity(name="São Jorge Castle", cost_per_person=15, travel_minutes_from_previous=10)])
            for i in range(1, 6)], food_per_person_per_day=40),
        budget=final, iterations=2, budget_history=[first, final],
        warnings=["Itinerary was planned around Chiado"], llm_tokens=123456,
    ).model_dump(mode="json")


def test_app_shows_form_prompt_without_a_plan():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    assert not at.exception
    assert "Plan my trip" in at.info[0].value


def test_app_renders_every_tab_for_a_plan():
    at = AppTest.from_file(str(APP), default_timeout=60)
    at.session_state["plan"] = sample_plan()
    at.run()

    assert not at.exception and not at.error
    assert [t.label for t in at.tabs] == ["Overview", "Flights", "Stay", "Itinerary", "Budget"]
    metrics = {m.label: m.value for m in at.metric}
    assert metrics["Total"] == "680 USD" and metrics["Status"] == "Within budget"
    assert metrics["Planning rounds"] == "2"
    assert len(at.expander) == 5  # one per itinerary day
    assert any("planned around Chiado" in w.value for w in at.warning)


def test_sample_button_loads_saved_plan():
    at = AppTest.from_file(str(APP), default_timeout=60).run()
    next(b for b in at.button if b.label == "Load sample result").click().run()
    assert not at.exception
    assert [t.label for t in at.tabs] == ["Overview", "Flights", "Stay", "Itinerary", "Budget"]
    assert any("Sample result" in c.value for c in at.caption)
