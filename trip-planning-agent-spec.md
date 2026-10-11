# Trip Planning Agent — Project Spec

## Overview
A multi-agent trip planning system built with **CrewAI**, using a hierarchical process where a manager agent coordinates specialist agents and handles budget feedback loops. Portfolio project for agentic AI / backend roles.

## Agents

1. **Destination Research Agent**
   - Pulls weather, visa requirements, best time to visit, local events
   - Tools: SerperDev/Tavily (search), OpenWeatherMap

2. **Flight/Transport Agent**
   - Searches flight options, budget estimates
   - Tools: Amadeus Self-Service API (free tier); scraping fallback if needed

3. **Accommodation Agent**
   - Hotel/stay options matching budget and preferences
   - Tools: SerperDev/Tavily (search), or hotel API if available

4. **Itinerary Planner Agent**
   - Sequences activities per day, accounts for geography/travel time
   - Tools: Google Maps Distance Matrix API or OpenRouteService (free)

5. **Budget Aggregator Agent**
   - Rolls up total cost, flags overage, suggests trims
   - Outputs structured JSON via Pydantic model
   - On overage: triggers feedback loop back to Flight/Accommodation agents to re-search cheaper options

## Architecture Notes
- Use `Process.hierarchical` in CrewAI so the manager agent coordinates agents and owns the budget feedback loop
- Tasks should use explicit `context=` dependencies (e.g., Itinerary Planner waits on Destination Research output)
- Budget feedback loop is the key differentiator — turns this from "agents calling tools" into "agents coordinating under constraints"

## Tech Stack

**Core**
- CrewAI (latest stable), `Process.hierarchical`
- LLM: Groq/LLaMA 3.3 70B (consistent with prior project) or GPT-4o-mini for stronger tool-use reliability

**Tools / APIs**
- Search: SerperDev or Tavily
- Weather: OpenWeatherMap (free tier)
- Flights: Amadeus Self-Service API (free tier) — flag as a known constraint since free flight APIs are limited
- Maps/distance: Google Maps Distance Matrix API or OpenRouteService (free, no billing card)
- Currency conversion: exchangerate-api.io (free tier)

**Backend**
- FastAPI — wraps the crew as an endpoint; input = destination, dates, budget, preferences
- Pydantic models for structured input/output (especially Budget Aggregator's JSON breakdown)

**Frontend**
- Streamlit — trip form in, itinerary + budget breakdown out

**Caching (optional)**
- SQLite or JSON cache for API responses (flights/hotels) to avoid burning quota during dev

## Why CrewAI (not AutoGen or LangGraph)
Trip planning is sequential/hierarchical delegation (flights → hotels → itinerary → budget check), not an iterative conversation needing convergence — CrewAI's role/task/process model maps directly onto this, unlike AutoGen's conversational GroupChat pattern or LangGraph's explicit state graphs.
