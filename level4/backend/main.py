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
import queue
import threading
import uuid

from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
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
def pipeline_status(include_outputs: bool = False):
    """Return current pipeline state for the monitoring dashboard."""
    return state.get_state().to_dict(include_outputs=include_outputs)


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
    s = state.get_state()
    if s.status == "running":
        return StreamingResponse(
            iter(["data: " + __import__("json").dumps({"type": "error", "error": "Pipeline already running"}) + "\n\n"]),
            media_type="text/event-stream",
        )

    session_id = request.session_id or str(uuid.uuid4())

    monitor.log_event(
        session_id=session_id,
        event_type="session_start",
        data={"question": request.question},
    )

    eq: queue.Queue[str | None] = queue.Queue(maxsize=2000)

    def _run_in_background():
        for event_str in orchestrator.run_pipeline(request.question, session_id):
            try:
                eq.put(event_str, block=False)
            except queue.Full:
                pass
        try:
            eq.put(None, block=False)
        except queue.Full:
            pass

    t = threading.Thread(target=_run_in_background, daemon=True)
    t.start()

    def _stream():
        while True:
            try:
                event = eq.get(timeout=30)
                if event is None:
                    break
                yield event
            except queue.Empty:
                yield ": keepalive\n\n"

    return StreamingResponse(
        _stream(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/stop")
def stop_pipeline():
    """Request cancellation of the running pipeline."""
    s = state.get_state()
    if s.status != "running":
        return {"status": "not_running"}
    state.request_cancel()
    return {"status": "stopping"}


@app.get("/events")
def events(session_id: str | None = None):
    return monitor.get_events(session_id=session_id)


@app.get("/sessions")
def list_sessions(limit: int = 20):
    return monitor.list_sessions(limit=limit)


@app.get("/sessions/{session_id}")
def get_session(session_id: str):
    session = monitor.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session
