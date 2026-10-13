"""Streamlit front end: trip form in, itinerary + budget breakdown out.

Run the API first (``uvicorn trip_planner.api:app``), then
``streamlit run app/streamlit_app.py``.
"""

from __future__ import annotations

import json
import os
import time
from datetime import date, timedelta
from pathlib import Path

import httpx
import pandas as pd
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")
POLL_SECONDS = 3
TIMEOUT_SECONDS = 15 * 60
# A real result saved from a live run, so the UI can be explored without
# spending LLM quota (a full hierarchical run uses most of a day's free tier).
SAMPLE_PLAN = Path(__file__).with_name("sample_plan.json")

st.set_page_config(page_title="Trip Planner", page_icon="✈️", layout="wide")
st.title("✈️ Trip Planning Agent")
st.caption("CrewAI hierarchical crew + budget feedback loop")

with st.sidebar.form("trip"):
    origin = st.text_input("From", "Mumbai")
    destination = st.text_input("To", "Lisbon")
    start = st.date_input("Start date", date.today() + timedelta(days=30))
    end = st.date_input("End date", date.today() + timedelta(days=35))
    travelers = st.number_input("Travellers", 1, 10, 1)
    budget = st.number_input("Total budget", min_value=1.0, max_value=None, value=1500.0, step=100.0)
    currency = st.selectbox("Currency", ["USD", "EUR", "INR", "GBP"])
    prefs = st.multiselect(
        "Preferences",
        ["food", "history", "museums", "nightlife", "nature", "beaches", "shopping", "budget-friendly"],
        ["food", "history"],
    )
    submitted = st.form_submit_button("Plan my trip", type="primary")

if SAMPLE_PLAN.exists() and st.sidebar.button("Load sample result", help="Saved live run, no API call"):
    st.session_state["plan"] = json.loads(SAMPLE_PLAN.read_text(encoding="utf-8"))
    st.session_state["is_sample"] = True


def run_plan(payload: dict) -> dict:
    with httpx.Client(base_url=API_URL, timeout=30) as c:
        job = c.post("/plan/async", json=payload).raise_for_status().json()
        started = time.time()
        with st.status("Agents are planning your trip…", expanded=False) as status:
            while time.time() - started < TIMEOUT_SECONDS:
                time.sleep(POLL_SECONDS)
                job = c.get(f"/plan/{job['id']}").raise_for_status().json()
                status.update(label=f"Agents are planning your trip… ({job['status']}, {int(time.time() - started)}s)")
                if job["status"] in {"done", "failed"}:
                    break
            if job["status"] != "done":
                status.update(label="Planning failed", state="error")
                raise RuntimeError(job.get("error") or "Timed out")
            status.update(label="Plan ready", state="complete")
    return job["result"]


if submitted:
    if end <= start:
        st.error("End date must be after start date.")
    else:
        payload = {
            "origin": origin, "destination": destination,
            "start_date": start.isoformat(), "end_date": end.isoformat(),
            "travelers": int(travelers), "budget": budget, "currency": currency,
            "preferences": prefs,
        }
        try:
            st.session_state["plan"] = run_plan(payload)
            st.session_state["is_sample"] = False
        except (httpx.HTTPError, RuntimeError) as e:
            st.error(f"Could not plan trip: {e}")

plan = st.session_state.get("plan")
if not plan:
    st.info("Fill in the trip details in the sidebar and press **Plan my trip**, "
            "or press **Load sample result** to see a saved run instantly.")
    st.stop()
if st.session_state.get("is_sample"):
    req = plan["request"]
    st.caption(f"Sample result from a saved live run: {req['origin']} → {req['destination']}, "
               f"{req['start_date']} to {req['end_date']}, budget {req['budget']:,.0f} {req['currency']}.")

b, cur = plan["budget"], plan["budget"]["currency"]
c1, c2, c3, c4 = st.columns(4)
c1.metric("Total", f"{b['total']:,.0f} {cur}")
c2.metric("Budget", f"{b['budget']:,.0f} {cur}")
c3.metric("Status", "Over budget" if b["over_budget"] else "Within budget",
          f"-{b['overage']:,.0f}" if b["over_budget"] else None)
c4.metric("Planning rounds", plan["iterations"],
          help=f"{plan.get('llm_tokens', 0):,} LLM tokens used" if plan.get("llm_tokens") else None)
for w in plan.get("warnings", []):
    st.warning(w)

overview, flights, stay, days, money = st.tabs(["Overview", "Flights", "Stay", "Itinerary", "Budget"])

with overview:
    d = plan.get("destination") or {}
    st.subheader(plan["request"]["destination"])
    st.write(d.get("summary", "—"))
    col1, col2 = st.columns(2)
    col1.markdown(f"**Weather:** {d.get('weather_summary') or '—'}")
    col1.markdown(f"**Visa:** {d.get('visa_requirements') or '—'}")
    col1.markdown(f"**Best time to visit:** {d.get('best_time_to_visit') or '—'}")
    col2.markdown("**Events during your trip**\n" + "\n".join(f"- {e}" for e in d.get("local_events", [])))
    col2.markdown("**Neighbourhoods**\n" + "\n".join(f"- {n}" for n in d.get("neighborhoods", [])))
    if d.get("tips"):
        st.markdown("**Tips**\n" + "\n".join(f"- {t}" for t in d["tips"]))

with flights:
    if plan.get("chosen_flight"):
        f = plan["chosen_flight"]
        st.success(f"Selected: {f['airline']} {f['route']} — {f['price_per_person']:,.0f} {f['currency']}/person ({f['source']})")
    if plan["flights"]:
        st.dataframe(pd.DataFrame(plan["flights"]), width="stretch", hide_index=True)
    else:
        st.write("No flight options found.")

with stay:
    if plan.get("chosen_stay"):
        s = plan["chosen_stay"]
        st.success(f"Selected: {s['name']} ({s['neighborhood']}) — {s['price_per_night']:,.0f} {cur}/night")
    if plan["stays"]:
        st.dataframe(pd.DataFrame(plan["stays"]), width="stretch", hide_index=True)
    else:
        st.write("No stay options found.")

with days:
    itin = plan.get("itinerary") or {"days": []}
    for day in itin["days"]:
        with st.expander(f"Day {day['day']} {day.get('date', '')} — {day.get('theme', '')}", expanded=day["day"] == 1):
            for a in day["activities"]:
                travel = f" · 🚶 {a['travel_minutes_from_previous']:.0f} min" if a.get("travel_minutes_from_previous") else ""
                cost = f" · {a['cost_per_person']:,.0f} {cur}" if a.get("cost_per_person") else " · free"
                st.markdown(f"**{a.get('time', '')}** {a['name']} — {a.get('location', '')}{cost}{travel}")
                if a.get("notes"):
                    st.caption(a["notes"])

with money:
    left, right = st.columns(2)
    breakdown = {k: b[k] for k in ["flights", "accommodation", "activities", "food", "local_transport"]}
    left.subheader("Breakdown")
    left.bar_chart(pd.Series(breakdown, name=cur))
    right.subheader("Budget feedback loop")
    history = pd.DataFrame(
        [{"round": i + 1, "total": h["total"], "budget": h["budget"]} for i, h in enumerate(plan["budget_history"])]
    ).set_index("round")
    right.line_chart(history)
    right.caption("Each round after the first re-searched cheaper flights and stays.")
    if b["suggested_trims"]:
        st.markdown("**Suggested trims**\n" + "\n".join(f"- {t}" for t in b["suggested_trims"]))
