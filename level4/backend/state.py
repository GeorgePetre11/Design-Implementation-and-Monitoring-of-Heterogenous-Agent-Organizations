"""
Level 4 -- In-memory pipeline state + WebSocket broadcast.

Tracks the current state of the hybrid pipeline so the monitoring
dashboard can display real-time agent status, including review and
revision states unique to Level 4.

Agent statuses: idle | working | done | reviewing | revising | approved | error
"""

import asyncio
import json
import time
from dataclasses import dataclass, field

from fastapi import WebSocket

_loop: asyncio.AbstractEventLoop | None = None


def set_event_loop(loop: asyncio.AbstractEventLoop):
    """Call once at app startup to store the main event loop."""
    global _loop
    _loop = loop


# ---------------------------------------------------------------------------
# Pipeline state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    name: str
    display_name: str
    model: str
    status: str = "idle"        # idle | working | done | reviewing | revising | approved | error
    elapsed: float | None = None
    output: dict | None = None
    error: str | None = None
    started_at: float | None = None
    revision_count: int = 0
    was_revised: bool = False


@dataclass
class PipelineState:
    session_id: str | None = None
    question: str | None = None
    status: str = "idle"        # idle | running | done | error
    started_at: float | None = None
    agents: dict[str, AgentState] = field(default_factory=dict)
    activity_log: list[dict] = field(default_factory=list)
    revisions: dict[str, dict] = field(default_factory=dict)

    def to_dict(self) -> dict:
        elapsed = None
        if self.started_at:
            elapsed = round(time.time() - self.started_at, 2)
        return {
            "session_id": self.session_id,
            "question": self.question,
            "status": self.status,
            "elapsed": elapsed,
            "agents": {
                k: {
                    "name": v.name,
                    "display_name": v.display_name,
                    "model": v.model,
                    "status": v.status,
                    "elapsed": v.elapsed,
                    "error": v.error,
                    "revision_count": v.revision_count,
                    "was_revised": v.was_revised,
                }
                for k, v in self.agents.items()
            },
            "activity_log": self.activity_log[-50:],
            "revisions": self.revisions,
        }


# Global mutable state
_state = PipelineState()


def get_state() -> PipelineState:
    return _state


def reset_state(session_id: str, question: str, agent_configs: list[dict]):
    """Reset state for a new pipeline run."""
    global _state
    _state = PipelineState(
        session_id=session_id,
        question=question,
        status="running",
        started_at=time.time(),
        agents={
            cfg["name"]: AgentState(
                name=cfg["name"],
                display_name=cfg["display_name"],
                model=cfg["model"],
            )
            for cfg in agent_configs
        },
        activity_log=[],
    )
    _log_activity("pipeline_start", f"Pipeline started for session {session_id[:8]}...")
    _broadcast_state()


def update_agent(name: str, status: str, **kwargs):
    """Update an agent's state and broadcast."""
    agent = _state.agents.get(name)
    if not agent:
        return
    agent.status = status
    if status == "working":
        agent.started_at = time.time()
    for k, v in kwargs.items():
        if hasattr(agent, k):
            setattr(agent, k, v)

    labels = {
        "working": f"{agent.display_name} started ({agent.model})",
        "done": f"{agent.display_name} completed" + (f" in {agent.elapsed}s" if agent.elapsed else ""),
        "reviewing": f"EM reviewing {agent.display_name}'s output...",
        "revising": f"{agent.display_name} revising based on EM feedback...",
        "approved": f"{agent.display_name} approved by EM" + (
            f" (after {agent.revision_count} revision)" if agent.was_revised else ""
        ),
        "error": f"{agent.display_name} failed: {agent.error or 'unknown'}",
    }
    _log_activity(f"agent_{status}", labels.get(status, f"{agent.display_name}: {status}"))
    _broadcast_state()


def finish_pipeline(revisions: dict | None = None):
    """Mark pipeline as done."""
    _state.status = "done"
    if revisions:
        _state.revisions = revisions
    total_revisions = sum(1 for r in (revisions or {}).values() if r.get("was_revised"))
    _log_activity(
        "pipeline_done",
        f"Pipeline completed ({total_revisions} revision{'s' if total_revisions != 1 else ''} needed)",
    )
    _broadcast_state()


def fail_pipeline(error: str):
    """Mark pipeline as failed."""
    _state.status = "error"
    _log_activity("pipeline_error", f"Pipeline failed: {error}")
    _broadcast_state()


def _log_activity(event_type: str, message: str):
    _state.activity_log.append({
        "timestamp": time.time(),
        "type": event_type,
        "message": message,
    })


# ---------------------------------------------------------------------------
# WebSocket broadcast
# ---------------------------------------------------------------------------
_connections: set[WebSocket] = set()


async def ws_connect(ws: WebSocket):
    await ws.accept()
    _connections.add(ws)
    try:
        await ws.send_text(json.dumps({"type": "state", "data": _state.to_dict()}))
    except Exception:
        _connections.discard(ws)


async def ws_disconnect(ws: WebSocket):
    _connections.discard(ws)


def _broadcast_state():
    """Broadcast current state to all connected dashboard clients."""
    if not _connections or _loop is None:
        return
    payload = json.dumps({"type": "state", "data": _state.to_dict()})

    async def _send_all():
        stale: list[WebSocket] = []
        for ws in list(_connections):
            try:
                await ws.send_text(payload)
            except Exception:
                stale.append(ws)
        for ws in stale:
            _connections.discard(ws)

    future = asyncio.run_coroutine_threadsafe(_send_all(), _loop)
    try:
        future.result(timeout=2)
    except Exception:
        pass
