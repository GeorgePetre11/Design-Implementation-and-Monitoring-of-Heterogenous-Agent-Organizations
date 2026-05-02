"""
Level 4 -- Pydantic models for request/response schemas.

Defines the input request and the structured output schemas for each agent.
These schemas enforce output structure (constraint layer #3).

Level 4 adds ReviewResult and RevisionInfo models for the hybrid workflow.
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
    service_overlap_notes: str = ""
    source: str = ""


class CustomerSegment(BaseModel):
    segment: str
    description: str
    size_estimate: str
    relevance_to_client_service: str = ""


class MarketTrend(BaseModel):
    """A market trend with relevance justification (Level 4).

    Accepts either a structured dict or a plain string (legacy) — the prompt
    requests structured objects but old runs may still deliver strings.
    """
    trend: str
    source: str = ""
    relevance_to_client_service: str = ""


class KeyFinding(BaseModel):
    """A key research finding with relevance justification (Level 4)."""
    finding: str
    source: str = ""
    relevance_to_client_service: str = ""


class RegulatoryLandscape(BaseModel):
    tax_regime: str = ""
    labor_law: str = ""
    data_protection: str = ""
    business_registration: str = ""


class MarketAnalysis(BaseModel):
    market_overview: str
    market_size_and_growth: str
    key_competitors: list[Competitor]
    # market_trends and key_findings remain list[str]-compatible for backward
    # compatibility; the prompt now asks for structured entries but the
    # orchestrator doesn't re-validate, so we keep types permissive.
    market_trends: list
    customer_segments: list[CustomerSegment]
    regulatory_landscape: RegulatoryLandscape | None = None
    key_findings: list
    sources: list[str] = []


# ---------------------------------------------------------------------------
# Financial Analyst output
# ---------------------------------------------------------------------------

class CostEstimate(BaseModel):
    category: str
    amount: str
    timeframe: str
    notes: str


class RevenueProjection(BaseModel):
    scenario: str
    year_1: str
    year_2: str
    year_3: str
    assumptions: str


class FinancialAnalysis(BaseModel):
    executive_summary: str
    revenue_model_type: str = ""            # one_time | recurring_contract | hybrid
    revenue_model_justification: str = ""
    data_inputs_used: str = ""
    cost_estimates: list[CostEstimate]
    revenue_projections: list[RevenueProjection]
    roi_analysis: str
    break_even_timeline: str
    sensitivity_analysis: str
    cost_per_employee_warning: str = ""
    key_financial_risks: list[str]


# ---------------------------------------------------------------------------
# Risk Analyst output
# ---------------------------------------------------------------------------

class Risk(BaseModel):
    id: int
    title: str
    description: str
    category: str          # regulatory / market / operational / competitive / financial
    probability: str       # low / medium / high
    impact: str            # low / medium / high
    source: str = ""       # URL or data source (new in L4)
    mitigation_suggestion: str


class RiskAssessment(BaseModel):
    overall_risk_level: str    # low / medium / high / critical
    risk_summary: str
    risks: list[Risk]
    key_risk_factors: list[str]


# ---------------------------------------------------------------------------
# Evaluator output
# ---------------------------------------------------------------------------

class CriterionScore(BaseModel):
    score: int             # 1-10
    justification: str


class EvaluationScorecard(BaseModel):
    completeness: CriterionScore
    accuracy: CriterionScore
    coherence: CriterionScore
    structure: CriterionScore
    actionability: CriterionScore
    critical_depth: CriterionScore
    overall_score: float
    summary: str
    strengths: list[str]
    weaknesses: list[str]


# ---------------------------------------------------------------------------
# Level 4 -- Review & Revision models
# ---------------------------------------------------------------------------

class ReviewResult(BaseModel):
    """Result of an EM review of an agent's output."""
    approved: bool
    completeness_ok: bool = True
    sources_ok: bool = True
    no_hallucinations: bool = True
    consistency_ok: bool = True
    quality_ok: bool = True
    feedback: str = ""


class RevisionInfo(BaseModel):
    """Tracks whether an agent went through a revision cycle."""
    agent_name: str
    was_revised: bool = False
    review_feedback: str = ""
    revision_count: int = 0
