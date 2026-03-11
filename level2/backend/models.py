"""
Level 2 — Pydantic models for request/response schemas.

Defines the input request and the structured output schemas for each agent.
These schemas enforce output structure (constraint layer #3).
"""

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Request
# ---------------------------------------------------------------------------

class AnalysisRequest(BaseModel):
    question: str
    session_id: str | None = None


# ---------------------------------------------------------------------------
# Engagement Manager output
# ---------------------------------------------------------------------------

class Workstream(BaseModel):
    id: int
    title: str
    description: str
    key_questions: list[str]


class AnalysisPlan(BaseModel):
    business_question_summary: str
    workstreams: list[Workstream]


# ---------------------------------------------------------------------------
# Market Researcher output
# ---------------------------------------------------------------------------

class Competitor(BaseModel):
    name: str
    description: str
    market_position: str


class CustomerSegment(BaseModel):
    segment: str
    description: str
    size_estimate: str


class MarketAnalysis(BaseModel):
    market_overview: str
    market_size_and_growth: str
    key_competitors: list[Competitor]
    market_trends: list[str]
    customer_segments: list[CustomerSegment]
    key_findings: list[str]


# ---------------------------------------------------------------------------
# Evaluator output
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    score: int
    justification: str


class EvaluationScores(BaseModel):
    completeness: CriterionScore
    accuracy: CriterionScore
    coherence: CriterionScore
    structure: CriterionScore
    actionability: CriterionScore
    critical_depth: CriterionScore


class Evaluation(BaseModel):
    scores: EvaluationScores
    overall_score: float
    summary: str
