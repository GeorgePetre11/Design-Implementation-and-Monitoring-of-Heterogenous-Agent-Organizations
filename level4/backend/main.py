"""
Level 4 -- FastAPI backend.

Exposes:
  POST /analyze    -- runs the hybrid pipeline, streams events via SSE
  GET  /status     -- returns current pipeline state (for dashboard polling)
  WS   /ws         -- WebSocket for real-time dashboard updates
  GET  /events     -- returns monitoring events from SQLite
  GET  /health     -- health check with per-agent model info
"""

import asyncio
import uuid

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

import agents
import monitor
import orchestrator
import state
from models import AnalysisRequest

app = FastAPI(title="AI Consulting Firm — Level 4 (Hybrid)")


@app.on_event("startup")
async def _capture_event_loop():
    """Store the main asyncio loop so sync threads can schedule WS sends."""
    state.set_event_loop(asyncio.get_running_loop())


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "level": 4,
        "workflow": "hybrid_hierarchical",
        "agents": agents.AGENT_MODELS,
    }


@app.get("/status")
def pipeline_status():
    """Return current pipeline state for the monitoring dashboard."""
    return state.get_state().to_dict()


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    """WebSocket for real-time dashboard updates."""
    await state.ws_connect(ws)
    try:
        while True:
            await ws.receive_text()
    except WebSocketDisconnect:
        await state.ws_disconnect(ws)


@app.post("/analyze")
def analyze(request: AnalysisRequest):
    session_id = request.session_id or str(uuid.uuid4())

    monitor.log_event(
        session_id=session_id,
        event_type="session_start",
        data={"question": request.question},
    )

    return StreamingResponse(
        orchestrator.run_pipeline(request.question, session_id),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.get("/events")
def events(session_id: str | None = None):
    return monitor.get_events(session_id=session_id)
