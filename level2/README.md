# Level 2 — Four-Agent Sequential Pipeline

## Overview

Level 2 introduces **role specialization** — four heterogeneous agents collaborate
in a fixed sequential pipeline to produce a consulting report. Each agent has a
dedicated role, a specific LLM model, and strict constraints on what it can do.

This is the first level where **multiple agents** work together, demonstrating
the core thesis concept of heterogeneous agent organizations.

## Architecture

```
Client Question
      │
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Engagement  │───▶│    Market    │───▶│   Strategy   │───▶│  Evaluator   │
│   Manager    │    │  Researcher  │    │  Consultant  │    │              │
│   qwen3:8b   │    │  qwen3:14b   │    │  qwen3:32b   │    │  qwen3:14b   │
└──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘
   Analysis Plan      Market Analysis    Consulting Report    Evaluation
      (JSON)              (JSON)           (Markdown)         Scorecard (JSON)
```

## Agents

| Agent | Role | Model | Output |
|-------|------|-------|--------|
| **Engagement Manager** | Decomposes the question into workstreams | qwen3:8b | Analysis plan (JSON) |
| **Market Researcher** | Investigates market landscape | qwen3:14b | Market analysis (JSON) |
| **Strategy Consultant** | Synthesizes findings into a report | qwen3:32b | Consulting report (Markdown, streamed) |
| **Evaluator** | Scores the report on a rubric | qwen3:14b | Evaluation scorecard (JSON) |

## Constraint Enforcement

Agents are limited through four layers:

1. **System prompts** — each agent's prompt explicitly states what it can and cannot do
2. **Output schemas** — EM, MR, and Evaluator must return valid JSON in a defined structure
3. **Orchestrator routing** — the pure-Python orchestrator controls what data each agent sees:
   - EM sees only the question
   - MR sees the question + analysis plan (not other agents' data)
   - SC sees the question + analysis plan + market research (cannot search for new data)
   - Evaluator sees only the question + final report (not intermediate outputs)
4. **Validation** — JSON extraction with fallback parsing catches malformed output

## Data Flow

```
Question ──────────────────────────────────────────────► Engagement Manager
Question + Analysis Plan ──────────────────────────────► Market Researcher
Question + Analysis Plan + Market Research ────────────► Strategy Consultant
Question + Final Report ───────────────────────────────► Evaluator
```

## Monitoring

The system has two layers of monitoring:

### SQLite Event Log
Events are logged to SQLite with:
- `session_start` — pipeline begins, stores the question
- `agent_start` — each agent begins, stores model name
- `agent_complete` — each agent finishes, stores elapsed time
- `agent_error` — if an agent fails
- `session_complete` — pipeline finishes

### Real-time Dashboard
A **separate monitoring dashboard** (`/dashboard.html`) displays live pipeline status
independently of the main chat interface. It connects via **WebSocket** to receive
real-time state updates:

- **Pipeline visualization** — four agent nodes with live status indicators (idle → working → done) and directional arrows
- **Activity log** — scrolling timeline of every pipeline event with timestamps
- **Performance metrics** — session ID, pipeline status, total elapsed time, per-agent timing bars
- **Client question** — the current business question being analyzed

The main page (`/index.html`) is kept clean — it shows only the input, report, and
evaluation. No agent status cards or pipeline visualization.

## File Structure

```
level2/
├── docker-compose.yml        # Two services: backend + frontend (nginx)
├── Dockerfile                # Python 3.12 slim image
├── entrypoint.sh             # Waits for Ollama, starts uvicorn
├── README.md
├── backend/
│   ├── main.py               # FastAPI app — /analyze, /status, /ws, /events, /health
│   ├── agents.py             # Four agent classes with system prompts
│   ├── orchestrator.py       # Sequential pipeline — routes data between agents
│   ├── models.py             # Pydantic schemas for requests and agent outputs
│   ├── monitor.py            # SQLite event logging
│   ├── state.py              # In-memory pipeline state + WebSocket broadcast
│   └── requirements.txt
└── frontend/
    ├── index.html            # Main page — input, report, evaluation (clean UI)
    └── dashboard.html        # Monitoring dashboard — pipeline viz, activity log, metrics
```

## Models

Models are recommended by the compass artifact analysis (Qwen 3 family via Ollama):

- **qwen3:8b** — fast structured decomposition (Engagement Manager)
- **qwen3:14b** — broad knowledge and synthesis (Market Researcher, Evaluator)
- **qwen3:32b** — superior writing and nuanced argumentation (Strategy Consultant)

All models are configurable via environment variables in `docker-compose.yml`.

## Running

### Prerequisites

- Docker and Docker Compose
- Ollama running on the host with the required models

### Pull the models

```bash
ollama pull qwen3:8b
ollama pull qwen3:14b
ollama pull qwen3:32b
```

### Start the system

```bash
cd level2
docker compose up --build
```

### Access the UI

| Page | URL | Description |
|------|-----|-------------|
| **Main page** | http://localhost:3000 | Input question, view report & evaluation |
| **Dashboard** | http://localhost:3000/dashboard.html | Real-time pipeline monitoring |

Open the dashboard in a separate browser tab before starting an analysis to
watch the agents work in real-time.

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Starts the four-agent pipeline, streams SSE events |
| `GET` | `/status` | Returns current pipeline state (JSON) |
| `WS` | `/ws` | WebSocket — pushes state updates to the dashboard |
| `GET` | `/events` | Returns monitoring events (optional `?session_id=`) |
| `GET` | `/health` | Returns status and model configuration |

## Comparison with Level 1

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Agents | 1 (generic) | 4 (specialized) |
| Models | 1 (qwen2.5:14b) | 3 different sizes (8b/14b/32b) |
| Role separation | None | Strict per-agent constraints |
| Output format | Free Markdown | JSON schemas + Markdown report |
| Orchestration | None | Sequential pipeline |
| Data routing | N/A | Controlled per-agent visibility |
| Evaluation | Self-evaluation | Independent evaluator agent |

## Thesis Context

Level 2 is the first step up from the baseline. It demonstrates that splitting
a single generalist agent into four specialized roles produces measurably
different — and expected to be higher quality — output. The progressive
complexity experiment compares Level 1 vs Level 2 evaluator scores on the
same business question to measure the impact of role specialization.
