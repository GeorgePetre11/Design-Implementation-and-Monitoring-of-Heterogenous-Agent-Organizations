# Level 2 — Three-Agent Sequential Pipeline with Web Search

## Overview

Level 2 introduces **role specialization** — three heterogeneous agents collaborate
in a fixed sequential pipeline to produce a consulting report. Each agent has a
dedicated role, a specific LLM model, and strict constraints on what it can do.

This is the first level where **multiple agents** work together, demonstrating
the core thesis concept of heterogeneous agent organizations. Compared to Level 1,
the key additions are role separation, controlled data routing between agents, and
real web search grounded in live data.

Evaluation is intentionally omitted at this level and handled externally.

## Architecture

```
Client Question
      │
      ▼
┌──────────────┐    ┌──────────────┐    ┌──────────────┐
│  Engagement  │───▶│    Market    │───▶│   Strategy   │
│   Manager    │    │  Researcher  │    │  Consultant  │
│   qwen3:8b   │    │  qwen3:14b   │    │  qwen3:32b   │
└──────────────┘    └──────────────┘    └──────────────┘
   Analysis Plan      Market Analysis    Consulting Report
      (JSON)              (JSON)           (Markdown, streamed)
```

## Agents

| Agent | Role | Model | Output |
|-------|------|-------|--------|
| **Engagement Manager** | Decomposes the question into workstreams | qwen3:8b | Analysis plan (JSON) |
| **Market Researcher** | Investigates market landscape via live web search | qwen3:14b | Market analysis (JSON) |
| **Strategy Consultant** | Synthesizes findings into a final report | qwen3:32b | Consulting report (Markdown, streamed) |

## Market Researcher — Web Search Tool Loop

The Market Researcher is the only agent with access to external tools. It operates
in two phases:

**Phase 1 — Research loop:** The LLM autonomously decides what to search for and
calls tools in a loop (up to 8 rounds) until it has gathered enough data.

```
LLM decides query → search_web(query) → results appended → LLM decides next action
                  → read_document(url) → page text appended → ... → loop ends
```

**Phase 2 — Synthesis:** A separate structured LLM call converts all collected
research into the required JSON schema.

### Tools

| Tool | Implementation | Description |
|------|---------------|-------------|
| `search_web(query, max_results)` | DuckDuckGo (`ddgs`) | Returns titles, URLs, and snippets. No API key required. |
| `read_document(url)` | `requests` + `BeautifulSoup` | Fetches a URL, strips boilerplate HTML, returns up to 3000 chars of clean text. |

Tool calls are logged to stdout for observability:
```
[tool:search_web] query='German B2B SaaS market size 2024' max_results=5
[tool:search_web] returned 5 results
[tool:read_document] url=https://...
[tool:read_document] fetched 8420 chars (truncated to 3000)
```

## Constraint Enforcement

Agents are limited through four layers:

1. **System prompts** — each agent's prompt explicitly states what it can and cannot do
2. **Output schemas** — EM and MR must return valid JSON in a defined structure
3. **Orchestrator routing** — the pure-Python orchestrator controls what data each agent sees:
   - EM sees only the question
   - MR sees the question + analysis plan (not other agents' data)
   - SC sees the question + analysis plan + market research (cannot search for new data)
4. **Validation** — JSON extraction with fallback parsing catches malformed output

## Data Flow

```
Question ──────────────────────────────────────────────► Engagement Manager
Question + Analysis Plan ──────────────────────────────► Market Researcher (+ web tools)
Question + Analysis Plan + Market Research ────────────► Strategy Consultant
```

## Monitoring

### SQLite Event Log
Events are logged to SQLite with:
- `session_start` — pipeline begins, stores the question
- `agent_start` — each agent begins, stores model name
- `agent_complete` — each agent finishes, stores elapsed time
- `agent_error` — if an agent fails
- `session_complete` — pipeline finishes

### Real-time Dashboard
A **separate monitoring dashboard** (`/dashboard.html`) displays live pipeline status
via WebSocket:

- **Pipeline visualization** — three agent nodes with live status indicators (idle → working → done)
- **Activity log** — scrolling timeline of every pipeline event with timestamps
- **Performance metrics** — session ID, pipeline status, total elapsed time, per-agent timing
- **Client question** — the current business question being analyzed

## File Structure

```
level2/
├── docker-compose.yml        # Two services: backend + frontend (nginx)
├── Dockerfile                # Python 3.12 slim image
├── entrypoint.sh             # Waits for Ollama, starts uvicorn
├── README.md
├── backend/
│   ├── main.py               # FastAPI app — /analyze, /status, /ws, /events, /health
│   ├── agents.py             # Three agent classes + web search tool implementations
│   ├── orchestrator.py       # Sequential pipeline — routes data between agents
│   ├── models.py             # Pydantic schemas for requests and agent outputs
│   ├── monitor.py            # SQLite event logging
│   ├── state.py              # In-memory pipeline state + WebSocket broadcast
│   └── requirements.txt
└── frontend/
    ├── index.html            # Main page — input and streaming report
    └── dashboard.html        # Monitoring dashboard — pipeline viz, activity log, metrics
```

## Dependencies

```
fastapi, uvicorn       — API server and SSE streaming
ollama                 — local LLM inference via Ollama
pydantic               — output schema validation
websockets             — real-time dashboard updates
ddgs                   — DuckDuckGo web search (no API key)
requests               — HTTP page fetching
beautifulsoup4         — HTML parsing and text extraction
```

## Models

All models run locally via Ollama (Qwen 3 family):

- **qwen3:8b** — fast structured decomposition (Engagement Manager, ~18s)
- **qwen3:14b** — broad knowledge, tool use, synthesis (Market Researcher, ~90s with searches)
- **qwen3:32b** — superior writing and reasoning (Strategy Consultant, ~600s)

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
| **Main page** | http://localhost:3000 | Input question, view streaming report |
| **Dashboard** | http://localhost:3000/dashboard.html | Real-time pipeline monitoring |

### API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Starts the pipeline, streams SSE events |
| `GET` | `/status` | Returns current pipeline state (JSON) |
| `WS` | `/ws` | WebSocket — pushes state updates to the dashboard |
| `GET` | `/events` | Returns monitoring events (optional `?session_id=`) |
| `GET` | `/health` | Returns status and model configuration |

### Check web search activity

```bash
docker logs level2-backend-1 2>&1 | grep "\[tool:"
```

## Comparison with Level 1

| Aspect | Level 1 | Level 2 |
|--------|---------|---------|
| Agents | 1 (generic) | 3 (specialized) |
| Models | 1 (qwen2.5:14b) | 3 different sizes (8b/14b/32b) |
| Role separation | None | Strict per-agent constraints |
| Output format | Free Markdown | JSON schemas + Markdown report |
| Orchestration | None | Sequential pipeline |
| Data routing | N/A | Controlled per-agent visibility |
| Web search | None | Live DuckDuckGo search (Market Researcher) |
| Evaluation | Self-evaluation in report | External (done separately) |

## Thesis Context

Level 2 is the first step up from the baseline. It demonstrates:
- **Role specialization** — splitting one generalist agent into three focused agents
- **Heterogeneity** — three different model sizes assigned by task complexity
- **Grounded research** — the Market Researcher uses live web search instead of relying
  solely on training data, producing more current and specific market intelligence
- **Controlled data flow** — each agent sees only what it needs, enforced by the orchestrator

The progressive complexity experiment runs the same business question through Level 1
and Level 2 and compares the output quality, with evaluation done externally.
