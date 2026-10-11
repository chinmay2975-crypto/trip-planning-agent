"""Pydantic contracts shared by the crew (output_pydantic), the flow and the API.

Models are kept flat on purpose: smaller open models (LLaMA 3.3) produce valid
JSON far more reliably for shallow schemas.
"""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field, field_validator, model_validator


class TripRequest(BaseModel):
    origin: str = Field(..., description="Departure city, e.g. 'Mumbai'")
    destination: str = Field(..., description="Destination city, e.g. 'Lisbon'")
    start_date: date
    end_date: date
    travelers: int = Field(1, ge=1, le=10)
    budget: float = Field(..., gt=0, description="Total trip budget for all travelers")
    currency: str = Field("USD", min_length=3, max_length=3)
    preferences: list[str] = Field(default_factory=list, examples=[["food", "museums"]])

    @model_validator(mode="after")
    def _check_dates(self) -> TripRequest:
        if self.end_date <= self.start_date:
            raise ValueError("end_date must be after start_date")
        self.currency = self.currency.upper()
        return self

    @property
    def nights(self) -> int:
        return (self.end_date - self.start_date).days

    @property
    def days(self) -> int:
        return self.nights + 1


class DestinationInfo(BaseModel):
    summary: str
    best_time_to_visit: str = ""
    visa_requirements: str = ""
    weather_summary: str = ""
    local_events: list[str] = Field(default_factory=list)
    neighborhoods: list[str] = Field(default_factory=list)
    tips: list[str] = Field(default_factory=list)


class FlightOption(BaseModel):
    airline: str
    route: str = Field(..., description="e.g. 'BOM-LIS-BOM'")
    outbound: str = ""
    inbound: str = ""
    stops: int = 0
    price_per_person: float = Field(..., ge=0)
    currency: str = "USD"
    source: str = Field("", description="'amadeus' or 'estimate' plus URL if any")

    @field_validator("outbound", "inbound", "source", mode="before")
    @classmethod
    def _coerce_text(cls, v: object) -> str:
        # Models sometimes return these as nested objects ({"date": ..., "time": ...});
        # one such field would otherwise fail validation of the whole option list.
        if v is None:
            return ""
        if isinstance(v, dict):
            return " ".join(str(x) for x in v.values() if x not in (None, ""))
        return str(v)


class FlightOptions(BaseModel):
    options: list[FlightOption] = Field(default_factory=list)


class StayOption(BaseModel):
    name: str
    neighborhood: str = ""
    price_per_night: float = Field(..., ge=0, description="Per room/unit for the whole group")
    rating: float | None = None
    url: str = ""
    notes: str = ""


class StayOptions(BaseModel):
    options: list[StayOption] = Field(default_factory=list)


class Activity(BaseModel):
    time: str = Field("", description="e.g. 'Morning' or '10:00'")
    name: str
    location: str = ""
    cost_per_person: float = 0.0
    travel_minutes_from_previous: float | None = None
    notes: str = ""


class DayPlan(BaseModel):
    day: int
    date: str = ""
    theme: str = ""
    activities: list[Activity] = Field(default_factory=list)


class Itinerary(BaseModel):
    days: list[DayPlan] = Field(default_factory=list)
    food_per_person_per_day: float = 0.0
    local_transport_per_day: float = Field(0.0, description="For the whole group")


class BudgetReport(BaseModel):
    currency: str = "USD"
    budget: float
    flights: float = 0.0
    accommodation: float = 0.0
    activities: float = 0.0
    food: float = 0.0
    local_transport: float = 0.0
    total: float = 0.0
    over_budget: bool = False
    overage: float = 0.0
    suggested_trims: list[str] = Field(default_factory=list)

    def breakdown(self) -> dict[str, float]:
        return {
            "flights": self.flights,
            "accommodation": self.accommodation,
            "activities": self.activities,
            "food": self.food,
            "local_transport": self.local_transport,
        }


class TripPlan(BaseModel):
    request: TripRequest
    destination: DestinationInfo | None = None
    flights: list[FlightOption] = Field(default_factory=list)
    stays: list[StayOption] = Field(default_factory=list)
    chosen_flight: FlightOption | None = None
    chosen_stay: StayOption | None = None
    itinerary: Itinerary | None = None
    budget: BudgetReport
    iterations: int = 1
    budget_history: list[BudgetReport] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    llm_tokens: int = Field(0, description="Total LLM tokens used across all crew runs")
