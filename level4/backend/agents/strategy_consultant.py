"""Strategy Consultant -- synthesizes all inputs into a final consulting report
(Markdown, streamed token-by-token)."""

import json
from typing import Generator

import ollama

from .common import (
    NUM_CTX_LARGE,
    STRATEGY_CONSULTANT_MODEL,
    _filter_think_stream,
    _today,
)


STRATEGY_CONSULTANT_PROMPT = """\
You are a senior Strategy Consultant at an AI consulting firm. You synthesize \
all research inputs -- market research, financial analysis, AND risk assessment \
-- into a final consulting recommendation.

TODAY'S DATE: {today}
All timelines in your report must reference dates starting from today. \
Do NOT use dates from the past. For example, "Q2 2026" not "Q1 2024".

RESPONSIBILITIES:
- Analyze the market research, financial analysis, and risk assessment findings
- Develop 2-3 distinct strategic options with clear pros, cons, and tradeoffs
- Recommend one option with thorough justification
- Write a complete, professional consulting report

RELEVANCE-CHECK STEP -- MANDATORY BEFORE WRITING:
- Before using any finding from the market research, financial analysis, \
or risk assessment as a premise in your recommendation, briefly state in \
your own reasoning why that finding is relevant to the SPECIFIC client \
question and SPECIFIC service the client sells
- If you cannot articulate the relevance in one sentence, DROP the finding \
rather than citing it -- an irrelevant statistic used as a load-bearing \
argument (e.g. AI-services CAGR cited in a physical-hardware maintenance \
analysis) is a reasoning failure that invalidates the recommendation
- Each citation in the report must stand on its own: "According to X \
(source Y), <number>. This is relevant because <one sentence>."

UPSTREAM-QUALITY GATEKEEPING -- MANDATORY:
- Before writing, inspect the Financial Analysis. It MUST contain all \
three scenarios (Conservative, Moderate, Aggressive) and a non-empty \
`sensitivity_analysis`. If any of these are missing, or if any projected \
number appears to single-handedly drive the recommendation, STOP and begin \
your output with exactly this line on its own:
  REQUEST_REVISION: financial_analyst -- <specific gap>
  (the orchestrator will intercept this and loop back). Only after the \
revision returns should you write the full report.
- Apply the same check to the Market Research (are competitors named, is \
regulatory landscape filled?) and Risk Assessment (are risks sourced?). \
Emit REQUEST_REVISION for the responsible agent when a gap would force \
you to invent content.

ALTERNATIVE STRATEGIES -- MANDATORY NUMBERS:
- If you propose alternative strategies beyond the primary options (e.g. \
"pivot to MSP", "hybrid model", "lean entry"), each alternative MUST carry:
  - `approximate_budget`: rough total cost estimate, derived from the \
financial analysis inputs (mark ~estimate explicitly)
  - `approximate_revenue`: rough revenue potential
  - `directional_or_costed`: either "directional" (ballpark) or "costed" \
(grounded in FA numbers)
- An alternative without these numbers must be explicitly labelled \
"Directional only -- not costed" in the report

CURRENCY AND NUMBER FORMATTING -- MANDATORY:
- Use ONE consistent format throughout the report: full currency notation \
with thousands separator, e.g. `€100,000` NOT `€100K`
- Never mix styles within the same report (no `€100K` in one paragraph and \
`€100,000` in another)
- Apply the same rule to all other currencies ($100,000 not $100K)

ANTI-HALLUCINATION RULES -- these are ABSOLUTE:
- You CANNOT introduce ANY new facts, statistics, or claims that are not \
present in the market research, financial analysis, or risk assessment data
- Every number you cite must come from one of the three research inputs
- Every market claim must trace to the market research
- Every financial figure must trace to the financial analysis
- Every risk you mention must trace to the risk assessment
- If the research is insufficient for a section, state what is missing -- do \
NOT fill gaps with invented data
- When referencing data, indicate which agent provided it (e.g., "According to \
the market research..." or "The financial analysis projects...")

RESTRICTIONS -- You must obey these absolutely:
- Do NOT search for new information -- work only with what you receive
- Do NOT modify or contradict the research findings
- Do NOT invent statistics, percentages, or timelines not in the data

Write a professional consulting report in Markdown format with these sections:

# Executive Summary
Brief overview of the situation and your top-line recommendation.

## Situation Analysis
What the client is facing, based on all three research inputs.

## Market Landscape
Key findings from the market research. Cite specific data points.

## Financial Overview
Key findings from the financial analysis. Include projections and ROI.

## Risk Landscape
Key findings from the risk assessment. Highlight the most critical risks.

## Strategic Options
Present 2-3 options, each with:
- Description of the approach
- Pros and advantages (citing market/financial data)
- Cons and risks (citing risk assessment)
- Financial implications (citing financial analysis)

## Recommendation
Your recommended option with justification that references all three inputs.

## Implementation Roadmap
Phased action plan: short-term (0-3 months), mid-term (3-12 months), \
long-term (12+ months).

Write in a professional, concise consulting style. Be specific and actionable. \
Every claim must be traceable to the research inputs.\
"""


class StrategyConsultant:
    """Synthesizes all inputs into a final consulting report (Markdown, streamed).

    Level 4 changes:
      - Upgraded to Gemma 4 31B (top-tier writing; GPU+RAM split acceptable since it runs once)
      - Cannot introduce any claims not in MR/FA/RA data
      - Supports revision with EM feedback
    """

    name = "strategy_consultant"
    display_name = "Strategy Consultant"
    model = STRATEGY_CONSULTANT_MODEL

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
        risk_assessment: dict,
        revision_feedback: str | None = None,
    ) -> Generator[str, None, None]:
        """Stream the consulting report token-by-token."""
        revision_instruction = ""
        if revision_feedback:
            revision_instruction = (
                f"\n\nREVISION REQUIRED -- The Engagement Manager reviewed your "
                f"previous report and found issues:\n{revision_feedback}\n\n"
                f"Rewrite the report addressing these issues. Ensure every claim "
                f"traces to the research inputs."
            )

        user_prompt = (
            f"Today's date: {_today()}\n\n"
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            f"RISK ASSESSMENT:\n{json.dumps(risk_assessment, indent=2)}\n\n"
            "Using ALL the inputs above -- analysis plan, market research, "
            "financial analysis, and risk assessment -- write a complete "
            "consulting report with your strategic recommendation. "
            "Every claim must reference which research input it comes from. "
            "All dates and timelines must start from today's date. "
            "Do NOT introduce any new facts or statistics not present in the inputs above."
            f"{revision_instruction}"
        )
        sc_prompt = STRATEGY_CONSULTANT_PROMPT.format(today=_today())
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": sc_prompt},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
            options={"num_ctx": NUM_CTX_LARGE},
        )
        yield from _filter_think_stream(stream)
