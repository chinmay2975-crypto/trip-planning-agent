"""Crew definitions.

``TripCrew`` is the full hierarchical crew: a manager agent (``LLM_MODEL``)
runs every task and delegates to the five specialists
(``SPECIALIST_LLM_MODEL``). ``CREW_PROCESS=sequential`` drops the manager.
``ReplanCrew`` is the small crew the flow calls inside the budget feedback
loop to re-search cheaper flights and stays.
"""

from __future__ import annotations

from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task

from ..config import get_llm, get_settings, get_specialist_llm
from ..models import BudgetReport, DestinationInfo, FlightOptions, Itinerary, StayOptions
from ..tools import CurrencyTool, FlightSearchTool, RoutingTool, WebSearchTool, WeatherTool

# Low iteration caps keep a run inside free-tier LLM token quotas. Every
# manager iteration re-sends the whole conversation, so the manager's cap
# matters most.
AGENT_MAX_ITER = 5
MANAGER_MAX_ITER = 6


def _manager() -> Agent:
    return Agent(
        role="Trip Planning Manager",
        goal=(
            "Get each task done by the right specialist with as few delegations as "
            "possible, and return their result in the required format."
        ),
        backstory=(
            "An experienced travel operations lead. You delegate each task once, with "
            "all the details the specialist needs. If the answer has the required "
            "fields you accept it and pass it on as your final answer. You re-delegate "
            "only when required fields are missing, and never repeat the same request."
        ),
        llm=get_llm(),
        allow_delegation=True,
        max_iter=MANAGER_MAX_ITER,
        verbose=True,
    )


def _specialist(config: dict, tools: list) -> Agent:
    return Agent(
        config=config,
        tools=tools,
        llm=get_specialist_llm(),
        allow_delegation=False,
        max_iter=AGENT_MAX_ITER,
        verbose=True,
    )


@CrewBase
class TripCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def destination_researcher(self) -> Agent:
        return _specialist(self.agents_config["destination_researcher"], [WebSearchTool(), WeatherTool()])

    @agent
    def flight_agent(self) -> Agent:
        return _specialist(self.agents_config["flight_agent"], [FlightSearchTool(), CurrencyTool(), WebSearchTool()])

    @agent
    def accommodation_agent(self) -> Agent:
        return _specialist(self.agents_config["accommodation_agent"], [WebSearchTool(), CurrencyTool()])

    @agent
    def itinerary_planner(self) -> Agent:
        return _specialist(self.agents_config["itinerary_planner"], [RoutingTool(), WebSearchTool()])

    @agent
    def budget_aggregator(self) -> Agent:
        return _specialist(self.agents_config["budget_aggregator"], [CurrencyTool()])

    @task
    def research_task(self) -> Task:
        return Task(config=self.tasks_config["research_task"], output_pydantic=DestinationInfo)

    @task
    def flight_task(self) -> Task:
        return Task(config=self.tasks_config["flight_task"], output_pydantic=FlightOptions)

    @task
    def stay_task(self) -> Task:
        return Task(config=self.tasks_config["stay_task"], output_pydantic=StayOptions)

    @task
    def itinerary_task(self) -> Task:
        return Task(config=self.tasks_config["itinerary_task"], output_pydantic=Itinerary)

    @task
    def budget_task(self) -> Task:
        return Task(config=self.tasks_config["budget_task"], output_pydantic=BudgetReport)

    @crew
    def crew(self) -> Crew:
        s = get_settings()
        if s.crew_process == "sequential":
            # Cheap dev mode: no manager, each task goes straight to its agent.
            return Crew(agents=self.agents, tasks=self.tasks, process=Process.sequential,
                        max_rpm=s.max_rpm, verbose=True)
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.hierarchical,
            manager_agent=_manager(),
            max_rpm=s.max_rpm,
            verbose=True,
        )


@CrewBase
class ReplanCrew:
    agents_config = "config/agents.yaml"
    tasks_config = "config/replan_tasks.yaml"

    @agent
    def flight_agent(self) -> Agent:
        return _specialist(self.agents_config["flight_agent"], [FlightSearchTool(), CurrencyTool(), WebSearchTool()])

    @agent
    def accommodation_agent(self) -> Agent:
        return _specialist(self.agents_config["accommodation_agent"], [WebSearchTool(), CurrencyTool()])

    @task
    def cheaper_flight_task(self) -> Task:
        return Task(config=self.tasks_config["cheaper_flight_task"], output_pydantic=FlightOptions)

    @task
    def cheaper_stay_task(self) -> Task:
        return Task(config=self.tasks_config["cheaper_stay_task"], output_pydantic=StayOptions)

    @crew
    def crew(self) -> Crew:
        # Two independent searches: sequential is enough, no manager overhead.
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            max_rpm=get_settings().max_rpm,
            verbose=True,
        )
