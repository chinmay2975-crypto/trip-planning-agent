from types import SimpleNamespace as NS

from trip_planner.flow import TripPlanningFlow, _parse
from trip_planner.models import (BudgetReport, DestinationInfo, FlightOption, FlightOptions,
                                 Itinerary, StayOption, StayOptions)


def trip_runner(flight_price: float, stay_price: float, agent_total: float | None = None):
    def run(inputs):
        budget = NS(pydantic=BudgetReport(budget=inputs["budget"], total=agent_total or 0,
                                          suggested_trims=["take a cheaper flight"]))
        return [
            NS(pydantic=DestinationInfo(summary="Lisbon")),
            NS(pydantic=FlightOptions(options=[FlightOption(airline="A", route="r", price_per_person=flight_price)])),
            NS(pydantic=StayOptions(options=[StayOption(name="H", price_per_night=stay_price)])),
            NS(pydantic=Itinerary(food_per_person_per_day=0)),
            budget,
        ]
    return run


def replan_runner(calls: list, factor: float = 1.0):
    """Return options priced at factor x the caps the flow asked for."""
    def run(inputs):
        calls.append(inputs)
        return [
            NS(pydantic=FlightOptions(options=[FlightOption(
                airline="B", route="r", price_per_person=inputs["max_flight_price"] * factor)])),
            NS(pydantic=StayOptions(options=[StayOption(
                name="G", price_per_night=inputs["max_stay_price"] * factor)])),
        ]
    return run


def run_flow(req, trip, replan):
    flow = TripPlanningFlow(request=req, trip_runner=trip, replan_runner=replan)
    flow.kickoff()
    return flow.state.result


def test_under_budget_finalizes_without_replan(request_1000):
    calls = []
    plan = run_flow(request_1000, trip_runner(400, 100), replan_runner(calls))
    assert plan.iterations == 1 and not calls
    assert plan.budget.total == 800 and not plan.budget.over_budget


def test_over_budget_triggers_replan_and_converges(request_1000):
    calls = []
    plan = run_flow(request_1000, trip_runner(900, 150), replan_runner(calls))
    assert len(calls) == 1
    assert plan.iterations == 2
    assert [round(b.total) for b in plan.budget_history] == [1500, 975]  # caps include a 5% safety margin
    assert not plan.budget.over_budget
    assert plan.chosen_flight.airline == "B" and plan.chosen_stay.name == "G"
    assert calls[0]["trims"] == "take a cheaper flight"


def test_stops_at_max_iterations_when_still_over(request_1000):
    calls = []

    # Replan options only ever get 10% cheaper, never enough.
    def stubborn(inputs):
        calls.append(inputs)
        f, s = inputs["current_flight_price"] * 0.9, inputs["current_stay_price"] * 0.9
        return [NS(pydantic=FlightOptions(options=[FlightOption(airline="B", route="r", price_per_person=f)])),
                NS(pydantic=StayOptions(options=[StayOption(name="G", price_per_night=s)]))]

    plan = run_flow(request_1000, trip_runner(2000, 300), stubborn)
    assert plan.iterations == 3 and len(calls) == 2
    assert plan.budget.over_budget
    totals = [b.total for b in plan.budget_history]
    assert totals == sorted(totals, reverse=True)
    assert any("over budget after 3" in w for w in plan.warnings)


def test_more_expensive_replan_results_are_ignored(request_1000):
    plan = run_flow(request_1000, trip_runner(900, 150), replan_runner([], factor=3))
    assert plan.chosen_flight.airline == "A" and plan.chosen_stay.name == "H"
    assert plan.iterations == 3


def test_agent_total_mismatch_is_warned(request_1000):
    plan = run_flow(request_1000, trip_runner(400, 100, agent_total=650), replan_runner([]))
    assert plan.budget.total == 800
    assert any("differed from computed" in w for w in plan.warnings)


def test_parse_falls_back_to_raw_json():
    out = NS(pydantic=None, raw='Here you go:\n{"summary": "Nice city"}\nThanks')
    assert _parse(out, DestinationInfo).summary == "Nice city"
    assert _parse(NS(pydantic=None, raw="no json"), DestinationInfo) is None


def test_estimated_flights_keep_only_dates(request_1000):
    from trip_planner.flow import normalize_flights
    est = FlightOption(airline="TK", route="r", outbound="TK724 02:30", inbound="TK725 14:20",
                       price_per_person=300, source="estimate: https://x")
    live = FlightOption(airline="TP", route="r", outbound="2026-11-01T09:00",
                        price_per_person=500, source="amadeus")
    out_est, out_live = normalize_flights([est, live], request_1000)
    assert (out_est.outbound, out_est.inbound) == ("2026-11-01", "2026-11-05")
    assert out_live.outbound == "2026-11-01T09:00"


def test_short_itinerary_is_warned(request_1000):
    from trip_planner.flow import itinerary_warnings
    from trip_planner.models import DayPlan
    short = Itinerary(days=[DayPlan(day=1), DayPlan(day=2), DayPlan(day=3)])
    assert itinerary_warnings(short, request_1000) == [
        "Itinerary covers 3 of 5 days; activity costs for the missing days are not included"]
    full = Itinerary(days=[DayPlan(day=i) for i in range(1, 6)])
    assert itinerary_warnings(full, request_1000) == []


def test_replan_to_other_neighbourhood_is_warned(request_1000):
    def trip(inputs):
        out = trip_runner(900, 150)(inputs)
        out[2] = NS(pydantic=StayOptions(options=[StayOption(name="H", neighborhood="Chiado", price_per_night=150)]))
        return out

    def replan(inputs):
        return [None, NS(pydantic=StayOptions(options=[
            StayOption(name="G", neighborhood="Arroios", price_per_night=40)]))]

    plan = run_flow(request_1000, trip, replan)
    assert plan.chosen_stay.name == "G"
    assert any("planned around Chiado" in w and "Arroios" in w for w in plan.warnings)


def test_token_usage_is_summed_across_crew_runs(request_1000):
    def with_usage(runner, tokens):
        return lambda inputs: NS(tasks_output=runner(inputs), token_usage=NS(total_tokens=tokens))

    plan = run_flow(request_1000, with_usage(trip_runner(900, 150), 5000),
                    with_usage(replan_runner([]), 1200))
    assert plan.iterations == 2 and plan.llm_tokens == 6200


def test_failed_replan_returns_best_plan_so_far(request_1000):
    def broken(inputs):
        raise RuntimeError("429 tokens per day")

    plan = run_flow(request_1000, trip_runner(900, 150), broken)
    assert plan is not None and plan.iterations == 1
    assert plan.budget.over_budget and plan.chosen_flight.airline == "A"
    assert any("Re-search for cheaper options failed" in w for w in plan.warnings)


def test_missing_flights_trigger_replan_even_when_under_budget(request_1000):
    # Live-run regression: round 1 lost its flights, so its total looked cheap.
    def no_flights(inputs):
        out = trip_runner(0, 100)(inputs)
        out[1] = NS(pydantic=FlightOptions(options=[]))
        return out

    calls = []
    plan = run_flow(request_1000, no_flights, replan_runner(calls))
    assert len(calls) == 1 and calls[0]["max_flight_price"] == 450.0
    assert plan.chosen_flight is not None and plan.budget.flights == 450.0
    assert plan.budget_history[0].flights == 0


def test_flights_never_found_is_warned(request_1000):
    def no_flights(inputs):
        out = trip_runner(0, 100)(inputs)
        out[1] = NS(pydantic=FlightOptions(options=[]))
        return out

    def stays_only(inputs):
        return [None, NS(pydantic=StayOptions(options=[StayOption(name="H2", price_per_night=90)]))]

    plan = run_flow(request_1000, no_flights, stays_only)
    assert plan.iterations == 3 and plan.chosen_flight is None
    assert any("No flights found after 3 planning rounds" in w for w in plan.warnings)
