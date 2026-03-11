# Level 1 — Single Agent (Baseline)

This is the simplest configuration of the AI Consulting Firm system. A single LLM-powered agent receives a business question and produces a full consulting report entirely by itself, with no role separation, no task delegation, and no inter-agent communication.

Its purpose in the thesis is to serve as the **baseline** against which all higher complexity levels are compared.

---

## How It Works

### 1. User Input
The user types a business question into the web interface (e.g. *"Should we expand into the US market?"*) and clicks **Analyze**.

### 2. Request Flow
```
Browser → POST /analyze → FastAPI backend → Ollama (qwen2.5:14b) → SSE stream → Browser
```

### 3. The Agent
The agent is a single call to `qwen2.5:14b` via Ollama. It is **not** a bare query — it has a detailed system prompt that instructs it to act as a senior business consultant and cover all analytical workstreams in one pass:

| # | Workstream | Description |
|---|-----------|-------------|
| 1 | Workstream Breakdown | Decomposes the question into key investigation areas |
| 2 | Market Analysis | Market size, growth, competitors, trends, customer segments |
| 3 | Financial Analysis | Cost estimates, revenue projections, ROI, break-even timeline |
| 4 | Risk Assessment | Top risks rated by probability and impact, with mitigations |
| 5 | Strategic Options & Recommendation | 2–3 options with pros/cons, one recommended with justification |
| 6 | Implementation Roadmap | Phased action plan (short / mid / long term) |
| 7 | Self-Evaluation | Agent scores its own report on 6 criteria (1–10 each) |

The system prompt enforces a structured Markdown output suitable for board-level presentation.

### 4. Streaming
The response is streamed token-by-token via **Server-Sent Events (SSE)**. The frontend renders the report live as it is generated.

### 5. Monitoring
Every run is logged to a local **SQLite database** (`monitoring.db`) with the following events:

| Event | When |
|-------|------|
| `session_start` | User submits a question |
| `agent_start` | Agent begins generating |
| `agent_complete` | Agent finishes, output length recorded |
| `session_complete` | Full session done |
| `agent_error` | If anything goes wrong |

Each event stores: `session_id`, `level`, `timestamp`, `event_type`, `agent_name`, `data`. This schema is shared across all levels.

### 6. Output
The report is displayed live in the browser. The user can:
- **Copy** the raw Markdown text to clipboard
- **Download PDF** — opens a print-ready page and triggers the browser's Save as PDF dialog

---

## Architecture

```
level1/
├── backend/
│   ├── main.py        # FastAPI server — /analyze (SSE), /events, /health
│   ├── agent.py       # Single agent: system prompt + Ollama call + streaming
│   ├── monitor.py     # SQLite event logger (reusable across all levels)
│   ├── models.py      # Pydantic request schema
│   └── requirements.txt
├── frontend/
│   └── index.html     # Single-page UI — input form, live report, metrics bar
├── Dockerfile         # Python 3.12 slim image for the backend
├── docker-compose.yml # Orchestrates backend + nginx frontend
├── entrypoint.sh      # Waits for Ollama, then starts uvicorn
└── .dockerignore
```

---

## Model

| Property | Value |
|----------|-------|
| Model | `qwen2.5:14b` |
| Provider | Ollama (local, open-source, free) |
| Why this model | Best balance of structured writing quality and business reasoning at this size among open-weight models |
| Configurable | Yes — set `OLLAMA_MODEL` env var to use a different model |

> **Note:** `qwen2.5:14b` requires ~9 GB of RAM. If your machine cannot fit it, use `qwen2.5:7b` or `qwen2.5:3b` by changing `OLLAMA_MODEL` in `docker-compose.yml`.

---

## Running

**Prerequisites:** Docker Desktop running, Ollama installed natively on your machine.

```bash
# 1. Pull the model on your host machine (one-time, ~9 GB)
ollama pull qwen2.5:14b

# 2. Start the system
cd level1
docker compose up --build

# 3. Open the UI
open http://localhost:3000
```

The backend runs on `http://localhost:8000` and connects to your host Ollama instance via `host.docker.internal:11434`, bypassing Docker's memory limits.

To stop:
```bash
docker compose down
```

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/analyze` | Start analysis — returns SSE stream of tokens |
| `GET` | `/events` | Retrieve monitoring events (optional `?session_id=`) |
| `GET` | `/health` | Health check — returns status and active model name |

---

## Thesis Context

This level is the **baseline** in the progressive complexity experiment. It demonstrates what a single generalist LLM can produce without any organizational structure, role specialization, or task decomposition. Its Evaluator scores (section 7 of each report) are compared against the outputs of Levels 2–5 to quantify the improvement gained by introducing specialization and organizational workflows.

**Key limitation:** The agent has no tools — it cannot search the web or access real data. All analysis is based solely on the model's parametric knowledge.
