"""
Level 1 — FastAPI backend.
Exposes:
  POST /analyze   — streams the consulting report via SSE
  GET  /events    — returns monitoring events from SQLite
  GET  /health    — basic health check
"""
import json
import uuid

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import agent
import monitor
from models import AnalysisRequest

app = FastAPI(title="AI Consulting Firm — Level 1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {"status": "ok", "level": 1, "model": agent.MODEL}


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    session_id = request.session_id or str(uuid.uuid4())

    monitor.log_event(
        session_id=session_id,
        event_type="session_start",
        data={"question": request.question},
    )
    monitor.log_event(
        session_id=session_id,
        event_type="agent_start",
        agent_name="SingleAgent",
        data={"model": agent.MODEL},
    )

    def event_stream():
        full_output = []
        try:
            for token in agent.run(request.question):
                full_output.append(token)
                payload = json.dumps({"token": token, "session_id": session_id})
                yield f"data: {payload}\n\n"

            monitor.log_event(
                session_id=session_id,
                event_type="agent_complete",
                agent_name="SingleAgent",
                data={"output_length": sum(len(t) for t in full_output)},
            )
            monitor.log_event(
                session_id=session_id,
                event_type="session_complete",
            )
            yield f"data: {json.dumps({'done': True, 'session_id': session_id})}\n\n"

        except Exception as e:
            monitor.log_event(
                session_id=session_id,
                event_type="agent_error",
                agent_name="SingleAgent",
                data={"error": str(e)},
            )
            yield f"data: {json.dumps({'error': str(e)})}\n\n"

    return StreamingResponse(
        event_stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/events")
def events(session_id: str | None = None):
    return monitor.get_events(session_id=session_id)
