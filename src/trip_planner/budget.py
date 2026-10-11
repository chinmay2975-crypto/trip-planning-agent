"""Deterministic budget math.

The Budget Aggregator agent writes the narrative (suggested trims), but the
numbers that drive the feedback loop are computed here: LLM arithmetic is not
something a router should trust.
"""

from __future__ import annotations

from .models import (
    BudgetReport,
    FlightOption,
    Itinerary,
    StayOption,
    TripRequest,
)


def cheapest_flight(options: list[FlightOption]) -> FlightOption | None:
    return min(options, key=lambda f: f.price_per_person, default=None)


def cheapest_stay(options: list[StayOption]) -> StayOption | None:
    return min(options, key=lambda s: s.price_per_night, default=None)


def compute_budget(
    request: TripRequest,
    flight: FlightOption | None,
    stay: StayOption | None,
    itinerary: Itinerary | None,
    suggested_trims: list[str] | None = None,
) -> BudgetReport:
    n = request.travelers
    flights = (flight.price_per_person * n) if flight else 0.0
    accommodation = (stay.price_per_night * request.nights) if stay else 0.0
    activities = food = local_transport = 0.0
    if itinerary:
        activities = sum(a.cost_per_person for d in itinerary.days for a in d.activities) * n
        food = itinerary.food_per_person_per_day * request.days * n
        local_transport = itinerary.local_transport_per_day * request.days

    total = round(flights + accommodation + activities + food + local_transport, 2)
    overage = round(max(0.0, total - request.budget), 2)
    return BudgetReport(
        currency=request.currency,
        budget=request.budget,
        flights=round(flights, 2),
        accommodation=round(accommodation, 2),
        activities=round(activities, 2),
        food=round(food, 2),
        local_transport=round(local_transport, 2),
        total=total,
        over_budget=overage > 0,
        overage=overage,
        suggested_trims=suggested_trims or [],
    )


def replan_caps(request: TripRequest, report: BudgetReport) -> tuple[float, float]:
    """Split the overage across flights and accommodation by their share.

    Returns (max flight price per person, max stay price per night). A 5%
    margin is added so the next iteration aims comfortably under budget.
    """
    flexible = report.flights + report.accommodation
    if flexible <= 0:
        return 0.0, 0.0
    cut = report.overage * 1.05
    flight_total = max(0.0, report.flights - cut * report.flights / flexible)
    stay_total = max(0.0, report.accommodation - cut * report.accommodation / flexible)
    per_person = flight_total / request.travelers
    per_night = stay_total / max(1, request.nights)
    return round(per_person, 2), round(per_night, 2)
