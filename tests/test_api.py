from fastapi.testclient import TestClient

from trip_planner import api
from trip_planner.models import BudgetReport, TripPlan, TripRequest

PAYLOAD = {"origin": "Mumbai", "destination": "Lisbon", "start_date": "2026-11-01",
           "end_date": "2026-11-05", "budget": 1000}


def fake_plan(req: TripRequest) -> TripPlan:
    return TripPlan(request=req, budget=BudgetReport(budget=req.budget, total=900))


def test_plan_endpoints(monkeypatch):
    monkeypatch.setattr(api, "plan_trip", fake_plan)
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}

    r = client.post("/plan", json=PAYLOAD)
    assert r.status_code == 200 and r.json()["budget"]["total"] == 900

    job = client.post("/plan/async", json=PAYLOAD).json()
    done = client.get(f"/plan/{job['id']}").json()  # TestClient runs background tasks inline
    assert done["status"] == "done" and done["result"]["budget"]["total"] == 900
    assert client.get("/plan/missing").status_code == 404


def test_invalid_request_rejected():
    client = TestClient(api.app)
    assert client.post("/plan", json=PAYLOAD | {"end_date": "2026-10-01"}).status_code == 422
