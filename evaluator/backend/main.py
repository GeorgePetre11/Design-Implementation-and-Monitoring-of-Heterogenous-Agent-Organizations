"""
Evaluator -- FastAPI backend.

Standalone service that scores a finished consulting report on six criteria
using Gemini 2.5 via Google AI Studio's OpenAI-compatible API. Other levels
(L2-L4) call this service after their own pipelines complete.

Endpoints:
  POST /evaluate   -- score a {question, report} pair
  GET  /events     -- monitoring events from SQLite
  GET  /health     -- health + model availability
"""
import time
import uuid

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

import monitor
from evaluator import DEFAULT_MODEL, EvaluatorAgent
from models import EvaluationRequest

app = FastAPI(title="AI Consulting Firm -- Evaluator (Gemini 2.5)")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single shared agent instance -- the OpenAI client is thread-safe.
_agent = EvaluatorAgent()


@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "evaluator",
        "model": _agent.model or DEFAULT_MODEL,
        "base_url": _agent.base_url,
        "available": _agent.available,
    }


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    if not _agent.available:
        raise HTTPException(
            status_code=503,
            detail="Evaluator unavailable: EVALUATOR_API_KEY is not configured.",
        )

    session_id = request.session_id or str(uuid.uuid4())

    monitor.log_event(
        session_id=session_id,
        event_type="evaluator_start",
        level=request.level,
        data={"model": _agent.model, "question_length": len(request.question or "")},
    )

    t0 = time.time()
    try:
        scorecard = _agent.evaluate(request.question, request.report)
    except RuntimeError as exc:
        monitor.log_event(
            session_id=session_id,
            event_type="evaluator_error",
            level=request.level,
            data={"error": str(exc)},
        )
        raise HTTPException(status_code=503, detail=str(exc))
    except ConnectionError as exc:
        monitor.log_event(
            session_id=session_id,
            event_type="evaluator_error",
            level=request.level,
            data={"error": str(exc)},
        )
        raise HTTPException(status_code=502, detail=str(exc))
    except ValueError as exc:
        monitor.log_event(
            session_id=session_id,
            event_type="evaluator_error",
            level=request.level,
            data={"error": str(exc)},
        )
        raise HTTPException(status_code=422, detail=str(exc))

    elapsed_ms = int((time.time() - t0) * 1000)

    monitor.log_event(
        session_id=session_id,
        event_type="evaluator_complete",
        level=request.level,
        data={
            "overall_score": scorecard.get("overall_score"),
            "latency_ms": elapsed_ms,
        },
    )

    return {
        "session_id": session_id,
        "model": _agent.model,
        "latency_ms": elapsed_ms,
        "scorecard": scorecard,
    }


@app.get("/events")
def events(session_id: str | None = None):
    return monitor.get_events(session_id=session_id)
