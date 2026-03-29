"""
Evaluator — Standalone FastAPI backend.

A separate application that evaluates consulting reports produced by any
level (L1–L4). Not integrated into the agent pipelines — runs independently.

Endpoints:
  POST /evaluate        — evaluate a report, returns scorecard JSON
  POST /evaluate/file   — evaluate a report from a saved file path
  GET  /results         — list all saved evaluation results
  GET  /results/{id}    — get a specific evaluation result
  GET  /health          — health check
"""

import json
import os
import time
import uuid
from pathlib import Path

import pdfplumber
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from evaluator import Evaluator, EVALUATOR_MODEL
from models import EvaluationRequest, EvaluationScorecard

app = FastAPI(title="AI Consulting Firm — Evaluator")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Storage — save evaluation results to JSON files
# ---------------------------------------------------------------------------

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def _save_result(evaluation_id: str, data: dict) -> Path:
    """Save an evaluation result to a JSON file."""
    path = RESULTS_DIR / f"{evaluation_id}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return path


# ---------------------------------------------------------------------------
# Request model for file-based evaluation
# ---------------------------------------------------------------------------

class FileEvaluationRequest(BaseModel):
    question: str
    report_path: str = Field(
        ..., description="Path to a Markdown/text file containing the report."
    )
    level: int = Field(..., ge=1, le=4)
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health")
def health():
    return {
        "status": "ok",
        "service": "evaluator",
        "model": EVALUATOR_MODEL,
    }


@app.post("/evaluate")
def evaluate(request: EvaluationRequest):
    """Evaluate a consulting report passed directly in the request body."""
    evaluation_id = request.session_id or str(uuid.uuid4())
    agent = Evaluator()

    print(
        f"[api] Starting evaluation {evaluation_id} "
        f"(level={request.level}, model={agent.model})",
        flush=True,
    )

    t0 = time.time()
    try:
        scorecard = agent.run(request.question, request.report, request.level)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    elapsed = round(time.time() - t0, 2)

    result = {
        "evaluation_id": evaluation_id,
        "level": request.level,
        "model": agent.model,
        "question": request.question,
        "elapsed_seconds": elapsed,
        "scorecard": scorecard,
    }

    _save_result(evaluation_id, result)
    print(
        f"[api] Evaluation {evaluation_id} complete in {elapsed}s — "
        f"overall_score={scorecard.get('overall_score')}",
        flush=True,
    )

    return result


@app.post("/evaluate/file")
def evaluate_file(request: FileEvaluationRequest):
    """Evaluate a consulting report loaded from a file on disk."""
    report_path = Path(request.report_path)
    if not report_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"Report file not found: {request.report_path}",
        )

    report = report_path.read_text(encoding="utf-8")
    if not report.strip():
        raise HTTPException(
            status_code=400,
            detail="Report file is empty.",
        )

    # Delegate to the main evaluate logic
    eval_request = EvaluationRequest(
        question=request.question,
        report=report,
        level=request.level,
        session_id=request.session_id,
    )
    return evaluate(eval_request)


@app.post("/evaluate/upload")
async def evaluate_upload(file: UploadFile = File(...)):
    """Evaluate a consulting report uploaded as a PDF or text file."""
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    contents = await file.read()
    suffix = Path(file.filename).suffix.lower()

    if suffix == ".pdf":
        import io
        try:
            with pdfplumber.open(io.BytesIO(contents)) as pdf:
                report = "\n\n".join(
                    page.extract_text() or "" for page in pdf.pages
                )
        except Exception as exc:
            raise HTTPException(
                status_code=400, detail=f"Failed to read PDF: {exc}"
            )
    elif suffix in (".md", ".txt", ".markdown"):
        report = contents.decode("utf-8", errors="replace")
    else:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Use .pdf, .md, or .txt",
        )

    report = report.strip()
    if not report:
        raise HTTPException(status_code=400, detail="File contains no text.")

    eval_request = EvaluationRequest(
        question="(extracted from report)",
        report=report,
        level=0,
    )
    return evaluate(eval_request)


@app.get("/results")
def list_results():
    """List all saved evaluation results, sorted by most recent."""
    results = []
    for path in sorted(RESULTS_DIR.glob("*.json"), reverse=True):
        try:
            with open(path, encoding="utf-8") as f:
                data = json.load(f)
            results.append({
                "evaluation_id": data.get("evaluation_id"),
                "level": data.get("level"),
                "elapsed_seconds": data.get("elapsed_seconds"),
                "overall_score": data.get("scorecard", {}).get("overall_score"),
                "question": data.get("question", "")[:100],
            })
        except Exception:
            continue
    return results


@app.get("/results/{evaluation_id}")
def get_result(evaluation_id: str):
    """Retrieve a specific evaluation result by ID."""
    path = RESULTS_DIR / f"{evaluation_id}.json"
    if not path.exists():
        raise HTTPException(status_code=404, detail="Evaluation not found.")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Serve frontend static files (must be LAST — catches all unmatched routes)
# ---------------------------------------------------------------------------

FRONTEND_DIR = Path(__file__).parent.parent / "frontend"
if FRONTEND_DIR.exists():
    from fastapi.responses import FileResponse

    @app.get("/")
    def serve_index():
        return FileResponse(FRONTEND_DIR / "index.html")

    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="frontend")
