"""
Evaluator -- Pydantic models.

The Evaluator scores a consulting report on six criteria (1-10 each)
and emits a structured scorecard. See README_evaluator_agent.md for the
full specification.
"""
from typing import List

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class EvaluationRequest(BaseModel):
    """Inputs accepted by the Evaluator. The agent receives ONLY these two
    fields -- no intermediate outputs, no agent logs, no metadata."""

    question: str = Field(..., description="The original client business question.")
    report: str = Field(..., description="The final consulting report to score.")
    session_id: str | None = Field(
        default=None,
        description="Optional caller-provided session ID for monitoring correlation.",
    )
    level: int | None = Field(
        default=None,
        description="Optional source level (1-5) for monitoring/analytics.",
    )


# ---------------------------------------------------------------------------
# Scorecard
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    score: int = Field(ge=1, le=10, description="Score from 1 (worst) to 10 (best)")
    justification: str = Field(
        min_length=30,
        max_length=800,
        description="2-4 sentence justification referencing specific report content",
    )


class EvaluationScorecard(BaseModel):
    """Strict schema matching the README specification."""

    evaluation: dict[str, CriterionScore] = Field(
        description=(
            "Scores for: completeness, accuracy, coherence, structure, "
            "actionability, critical_depth"
        ),
    )
    overall_score: float = Field(
        ge=1.0, le=10.0,
        description="Arithmetic mean of all 6 criterion scores, rounded to 1 decimal.",
    )
    summary: str = Field(
        min_length=30,
        max_length=600,
        description="1-3 sentence overall assessment.",
    )
    strongest_dimension: str = Field(description="The criterion with the highest score.")
    weakest_dimension: str = Field(description="The criterion with the lowest score.")
    critical_issues: List[str] = Field(
        default_factory=list,
        description="The 2-5 most severe problems found, if any.",
    )
