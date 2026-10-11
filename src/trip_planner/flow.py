"""TripPlanningFlow: the hierarchical crew wrapped in a budget feedback loop.

    plan ──► budget_check ──"finalize"──► finalize
                 ▲    │
                 │  "replan"
                 │    ▼
            replan_cheaper   (ReplanCrew: cheaper flights + stays under caps)

The router reads deterministic numbers from ``budget.compute_budget`` and stops
after ``MAX_BUDGET_ITERATIONS`` total planning rounds even if still over budget.
"""

from __future__ import annotations

import argparse
import json
import logging
from datetime import date, timedelta
from typing import Any, Callable, TypeVar

from crewai.flow.flow import Flow, listen, or_, router, start
from pydantic import BaseModel, Field, ValidationError

from .budget import cheapest_flight, cheapest_stay, compute_budget, replan_caps
from .config import get_settings
from .models import (
    BudgetReport,
    DestinationInfo,
    FlightOption,
    FlightOptions,
    Itinerary,
    StayOption,
    StayOptions,
    TripPlan,
    TripRequest,
)

log = logging.getLogger(__name__)
M = TypeVar("M", bound=BaseModel)


class TripState(BaseModel):
    request: TripRequest | None = None
    destination: DestinationInfo | None = None
    flights: list[FlightOption] = Field(default_factory=list)
    stays: list[StayOption] = Field(default_factory=list)
    chosen_flight: FlightOption | None = None
    chosen_stay: StayOption | None = None
    itinerary: Itinerary | None = None
    budget: BudgetReport | None = None
    iterations: int = 0
    budget_history: list[BudgetReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_tokens: int = 0
    replan_failed: bool = False
    result: TripPlan | None = None


def crew_inputs(req: TripRequest) -> dict[str, Any]:
    return {
        "origin": req.origin,
        "destination": req.destination,
        "start_date": req.start_date.isoformat(),
        "end_date": req.end_date.isoformat(),
        "travelers": req.travelers,
        "budget": req.budget,
        "currency": req.currency,
        "preferences": ", ".join(req.preferences) or "no particular preferences",
        "nights": req.nights,
        "days": req.days,
    }


def _parse(task_output: Any, model: type[M]) -> M | None:
    """Pydantic output if CrewAI parsed it, else try the raw text as JSON."""
    if task_output is None:
        return None
    if isinstance(getattr(task_output, "pydantic", None), model):
        return task_output.pydantic
    raw = (getattr(task_output, "raw", "") or "").strip()
    start_i, end_i = raw.find("{"), raw.rfind("}")
    if start_i == -1:
        return None
    try:
        return model.model_validate_json(raw[start_i : end_i + 1])
    except ValidationError:
        return None


# A runner returns a CrewOutput (tasks_output + token_usage) or, in tests, a
# plain list of task outputs.
CrewRunner = Callable[[dict[str, Any]], Any]


def run_trip_crew(inputs: dict[str, Any]) -> list[Any]:
    from .crew import TripCrew

    return TripCrew().crew().kickoff(inputs=inputs)


def run_replan_crew(inputs: dict[str, Any]) -> list[Any]:
    from .crew import ReplanCrew

    return ReplanCrew().crew().kickoff(inputs=inputs)


def normalize_flights(options: list[FlightOption], req: TripRequest) -> list[FlightOption]:
    """Estimated fares have no real schedule: keep only the travel dates.

    Models tend to invent flight numbers and times for estimates even when told
    not to, so this is enforced in code rather than trusted to the prompt.
    """
    return [
        f.model_copy(update={"outbound": req.start_date.isoformat(), "inbound": req.end_date.isoformat()})
        if not f.source.startswith("amadeus")
        else f
        for f in options
    ]


def itinerary_warnings(itinerary: Itinerary | None, req: TripRequest) -> list[str]:
    if not itinerary:
        return []
    planned = len(itinerary.days)
    if planned < req.days:
        return [f"Itinerary covers {planned} of {req.days} days; activity costs for the "
                "missing days are not included"]
    return []


class TripPlanningFlow(Flow[TripState]):
    # Crew runners are injected rather than overridden: CrewAI Flow subclasses
    # do not inherit @start/@listen methods, so tests pass stubs here instead.
    def __init__(
        self,
        request: TripRequest,
        trip_runner: CrewRunner = run_trip_crew,
        replan_runner: CrewRunner = run_replan_crew,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        self.state.request = request
        self._trip_runner = trip_runner
        self._replan_runner = replan_runner

    # --- flow steps ----------------------------------------------------------

    @start()
    def plan(self) -> None:
        req = self.state.request
        outputs = self._unwrap(self._trip_runner(crew_inputs(req)))
        outputs = list(outputs) + [None] * (5 - len(outputs))
        research, flights, stays, itinerary, budget = outputs[:5]

        self.state.destination = _parse(research, DestinationInfo)
        self.state.flights = normalize_flights((_parse(flights, FlightOptions) or FlightOptions()).options, req)
        self.state.stays = (_parse(stays, StayOptions) or StayOptions()).options
        self.state.itinerary = _parse(itinerary, Itinerary)
        agent_report = _parse(budget, BudgetReport)

        for name, value in [
            ("destination research", self.state.destination),
            ("flights", self.state.flights),
            ("stays", self.state.stays),
            ("itinerary", self.state.itinerary),
        ]:
            if not value:
                self.state.warnings.append(f"No usable {name} output from the crew")
        self.state.warnings += itinerary_warnings(self.state.itinerary, req)

        # Same selection rule the Budget Aggregator was told to use.
        self.state.chosen_flight = cheapest_flight(self.state.flights)
        self.state.chosen_stay = self.state.stays[0] if self.state.stays else None
        self._record_budget(agent_report.suggested_trims if agent_report else [])
        if agent_report and abs(agent_report.total - self.state.budget.total) > 1:
            self.state.warnings.append(
                f"Budget agent total {agent_report.total} differed from computed "
                f"{self.state.budget.total}; computed value used"
            )

    @router(or_(plan, "replan_cheaper"))
    def budget_check(self) -> str:
        b = self.state.budget
        if not b.over_budget:
            return "finalize"
        if self.state.replan_failed:
            return "finalize"
        if self.state.iterations >= get_settings().max_budget_iterations:
            self.state.warnings.append(
                f"Still {b.overage} {b.currency} over budget after "
                f"{self.state.iterations} planning rounds"
            )
            return "finalize"
        return "replan"

    @listen("replan")
    def replan_cheaper(self) -> None:
        req, b = self.state.request, self.state.budget
        max_flight, max_stay = replan_caps(req, b)
        inputs = crew_inputs(req) | {
            "overage": b.overage,
            "current_flight_price": self.state.chosen_flight.price_per_person if self.state.chosen_flight else "unknown",
            "current_stay_price": self.state.chosen_stay.price_per_night if self.state.chosen_stay else "unknown",
            "max_flight_price": max_flight,
            "max_stay_price": max_stay,
            "trims": "; ".join(b.suggested_trims) or "none",
        }
        try:
            flights_out, stays_out = (self._unwrap(self._replan_runner(inputs)) + [None, None])[:2]
        except Exception as e:
            # A failed re-search (e.g. LLM quota exhausted) must not throw away
            # the complete plan we already have: finish with it and say why.
            log.exception("Replan failed")
            self.state.replan_failed = True
            self.state.warnings.append(
                f"Re-search for cheaper options failed ({type(e).__name__}: {str(e)[:120]}); "
                "returning the best plan found so far"
            )
            return
        new_flights = normalize_flights((_parse(flights_out, FlightOptions) or FlightOptions()).options, req)
        new_stays = (_parse(stays_out, StayOptions) or StayOptions()).options

        # Only swap when the new option is actually cheaper, so totals never go up.
        f = cheapest_flight(new_flights)
        if f and (not self.state.chosen_flight or f.price_per_person < self.state.chosen_flight.price_per_person):
            self.state.chosen_flight = f
            self.state.flights = new_flights
        s = cheapest_stay(new_stays)
        if s and (not self.state.chosen_stay or s.price_per_night < self.state.chosen_stay.price_per_night):
            old = self.state.chosen_stay
            if old and old.neighborhood and s.neighborhood and old.neighborhood != s.neighborhood:
                self.state.warnings.append(
                    f"Itinerary was planned around {old.neighborhood}; the cheaper stay is in "
                    f"{s.neighborhood}, so check travel times from there"
                )
            self.state.chosen_stay = s
            self.state.stays = new_stays
        self._record_budget(b.suggested_trims)

    @listen("finalize")
    def build_plan(self) -> TripPlan:
        st = self.state
        st.result = TripPlan(
            request=st.request,
            destination=st.destination,
            flights=st.flights,
            stays=st.stays,
            chosen_flight=st.chosen_flight,
            chosen_stay=st.chosen_stay,
            itinerary=st.itinerary,
            budget=st.budget,
            iterations=st.iterations,
            budget_history=st.budget_history,
            warnings=st.warnings,
            llm_tokens=st.llm_tokens,
        )
        return st.result

    # --- helpers -------------------------------------------------------------

    def _unwrap(self, result: Any) -> list[Any]:
        """Task outputs from a runner result, adding up its LLM token usage."""
        usage = getattr(result, "token_usage", None)
        self.state.llm_tokens += getattr(usage, "total_tokens", 0) or 0
        return list(getattr(result, "tasks_output", result))

    def _record_budget(self, trims: list[str]) -> None:
        st = self.state
        st.budget = compute_budget(st.request, st.chosen_flight, st.chosen_stay, st.itinerary, trims)
        st.iterations += 1
        st.budget_history.append(st.budget)
        log.info("Round %d: total %.2f / budget %.2f", st.iterations, st.budget.total, st.budget.budget)


def plan_trip(request: TripRequest) -> TripPlan:
    flow = TripPlanningFlow(request=request)
    flow.kickoff()
    if flow.state.result is None:
        raise RuntimeError("Flow finished without a result")
    return flow.state.result


def main() -> None:
    p = argparse.ArgumentParser(description="Plan a trip from the command line")
    p.add_argument("--origin", default="Mumbai")
    p.add_argument("--destination", default="Lisbon")
    p.add_argument("--start", type=date.fromisoformat, default=date.today() + timedelta(days=30))
    p.add_argument("--nights", type=int, default=4)
    p.add_argument("--travelers", type=int, default=1)
    p.add_argument("--budget", type=float, default=1500)
    p.add_argument("--currency", default="USD")
    p.add_argument("--prefs", nargs="*", default=["food", "history"])
    a = p.parse_args()
    req = TripRequest(
        origin=a.origin,
        destination=a.destination,
        start_date=a.start,
        end_date=a.start + timedelta(days=a.nights),
        travelers=a.travelers,
        budget=a.budget,
        currency=a.currency,
        preferences=a.prefs,
    )
    plan = plan_trip(req)
    print(json.dumps(plan.model_dump(mode="json"), indent=2))


if __name__ == "__main__":
    main()
