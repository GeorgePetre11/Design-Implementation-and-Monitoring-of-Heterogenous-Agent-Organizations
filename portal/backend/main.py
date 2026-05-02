"""
Portal control backend.

Runs inside a container with access to the host Docker socket and the project
directory bind-mounted at the same absolute path as on the host. Exposes a
small HTTP API that the portal frontend calls to start/stop each level's
docker-compose stack without the user having to touch the terminal.
"""

import asyncio
import json
import os
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

PROJECT_ROOT = Path(os.environ["PROJECT_ROOT"]).resolve()

SERVICES: dict[str, Path] = {
    "level1": PROJECT_ROOT / "level1",
    "level2": PROJECT_ROOT / "level2",
    "level3": PROJECT_ROOT / "level3",
    "level4": PROJECT_ROOT / "level4",
    "evaluator": PROJECT_ROOT / "evaluator",
}

app = FastAPI(title="AI Consulting Firm — Portal Control")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


async def _run_compose(service: str, args: list[str]) -> dict:
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")

    service_dir = SERVICES[service]
    compose_file = service_dir / "docker-compose.yml"
    if not compose_file.exists():
        raise HTTPException(
            status_code=500,
            detail=f"compose file not found at {compose_file}",
        )

    cmd = ["docker", "compose", "-f", str(compose_file), *args]
    proc = await asyncio.create_subprocess_exec(
        *cmd,
        cwd=str(service_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    stdout, stderr = await proc.communicate()
    return {
        "service": service,
        "cmd": " ".join(cmd),
        "returncode": proc.returncode,
        "stdout": stdout.decode(errors="replace"),
        "stderr": stderr.decode(errors="replace"),
    }


async def _is_running(service: str) -> bool:
    """A service is 'running' if at least one of its compose containers is up."""
    result = await _run_compose(service, ["ps", "--format", "json"])
    if result["returncode"] != 0:
        return False

    # `docker compose ps --format json` emits either a JSON array (newer compose)
    # or one JSON object per line (older compose). Handle both.
    raw = result["stdout"].strip()
    if not raw:
        return False

    entries: list[dict] = []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            entries = parsed
        elif isinstance(parsed, dict):
            entries = [parsed]
    except json.JSONDecodeError:
        for line in raw.splitlines():
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    return any(entry.get("State") == "running" for entry in entries)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "project_root": str(PROJECT_ROOT),
        "services": list(SERVICES.keys()),
    }


@app.get("/api/services")
async def list_services():
    """Return running state for every managed service."""
    names = list(SERVICES.keys())
    states = await asyncio.gather(*[_is_running(name) for name in names])
    return {name: {"running": running} for name, running in zip(names, states)}


@app.get("/api/services/{service}")
async def get_service(service: str):
    if service not in SERVICES:
        raise HTTPException(status_code=404, detail=f"Unknown service: {service}")
    return {service: {"running": await _is_running(service)}}


@app.post("/api/services/{service}/start")
async def start_service(service: str):
    return await _run_compose(service, ["up", "-d", "--build"])


@app.post("/api/services/{service}/stop")
async def stop_service(service: str):
    return await _run_compose(service, ["down"])


@app.post("/api/services/start-all")
async def start_all():
    results = {}
    for name in SERVICES:
        results[name] = await _run_compose(name, ["up", "-d", "--build"])
    return results


@app.post("/api/services/stop-all")
async def stop_all():
    results = {}
    for name in SERVICES:
        results[name] = await _run_compose(name, ["down"])
    return results
