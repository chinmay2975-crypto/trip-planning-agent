"""FastAPI wrapper around TripPlanningFlow.

A full run takes minutes, so besides the blocking ``POST /plan`` there is an
async job API (``POST /plan/async`` + ``GET /plan/{job_id}``) the UI polls.
Jobs live in memory, which is fine for a single-process demo.
"""

from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime, timezone
from typing import Literal

from fastapi import BackgroundTasks, FastAPI, HTTPException
from pydantic import BaseModel

from .flow import plan_trip
from .models import TripPlan, TripRequest

log = logging.getLogger(__name__)
app = FastAPI(title="Trip Planning Agent", version="0.1.0")


class Job(BaseModel):
    id: str
    status: Literal["queued", "running", "done", "failed"] = "queued"
    created_at: datetime
    result: TripPlan | None = None
    error: str | None = None


JOBS: dict[str, Job] = {}


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/plan", response_model=TripPlan)
async def plan(request: TripRequest) -> TripPlan:
    try:
        return await asyncio.to_thread(plan_trip, request)
    except Exception as e:  # surface crew/LLM failures as a clean 502
        log.exception("Planning failed")
        raise HTTPException(status_code=502, detail=f"Planning failed: {e}") from e


def _run_job(job_id: str, request: TripRequest) -> None:
    job = JOBS[job_id]
    job.status = "running"
    try:
        job.result = plan_trip(request)
        job.status = "done"
    except Exception as e:
        log.exception("Job %s failed", job_id)
        job.error, job.status = str(e), "failed"


@app.post("/plan/async", response_model=Job, status_code=202)
def plan_async(request: TripRequest, background: BackgroundTasks) -> Job:
    job = Job(id=uuid.uuid4().hex, created_at=datetime.now(timezone.utc))
    JOBS[job.id] = job
    background.add_task(_run_job, job.id, request)
    return job


@app.get("/plan/{job_id}", response_model=Job)
def get_job(job_id: str) -> Job:
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job not found")
    return JOBS[job_id]
