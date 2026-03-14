"""
Evaluator — Pydantic models for input/output schemas.

The Evaluator is a standalone application that scores consulting reports
produced by any level (L1–L4) against a fixed rubric.
"""

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Input — what the evaluator receives
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    """Payload sent to the evaluator."""
    question: str = Field(
        ..., description="The original client business question."
    )
    report: str = Field(
        ..., description="The full consulting report (Markdown) to evaluate."
    )
    level: int = Field(
        ..., ge=1, le=4, description="Which complexity level produced this report."
    )
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Output — the evaluation scorecard
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    score: int = Field(..., ge=1, le=10, description="Score from 1 (worst) to 10 (best).")
    justification: str = Field(
        ..., description="2-3 sentence justification for this score."
    )


class EvaluationScorecard(BaseModel):
    """Structured evaluation output matching the thesis rubric."""
    completeness: CriterionScore
    accuracy: CriterionScore
    coherence: CriterionScore
    structure: CriterionScore
    actionability: CriterionScore
    critical_depth: CriterionScore
    overall_score: float = Field(
        ..., ge=1.0, le=10.0,
        description="Weighted average of all criterion scores.",
    )
    summary: str = Field(
        ..., description="2-3 sentence overall assessment of the report."
    )
    strengths: list[str] = Field(
        ..., description="Key strengths of the report (3-5 bullet points)."
    )
    weaknesses: list[str] = Field(
        ..., description="Key weaknesses or gaps (3-5 bullet points)."
    )
