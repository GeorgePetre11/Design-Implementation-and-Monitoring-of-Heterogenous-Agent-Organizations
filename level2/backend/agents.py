"""
Level 2 — Four Agents (Core Roles).

Four specialized agents collaborate in a sequential pipeline:
  1. Engagement Manager — decomposes the business question into workstreams
  2. Market Researcher  — investigates the market landscape
  3. Strategy Consultant — synthesizes findings into a consulting report
  4. Evaluator           — independently scores the final report

Each agent has:
  - A dedicated system prompt constraining its role  (soft constraint)
  - A specific LLM model — heterogeneous              (design choice)
  - A defined output schema — JSON or Markdown         (hard constraint)
  - Tool/data restrictions enforced by the orchestrator (hard constraint)

Models (from compass artifact — Qwen 3 family via Ollama):
  Engagement Manager  → qwen3:8b   (fast structured decomposition)
  Market Researcher   → qwen3:14b  (broad knowledge, synthesis)
  Strategy Consultant → qwen3:32b  (superior writing, reasoning)
  Evaluator           → qwen3:14b  (critical assessment)
"""

import json
import os
import re
from typing import Generator

import ollama

# ---------------------------------------------------------------------------
# Model configuration (override via environment variables)
# ---------------------------------------------------------------------------
ENGAGEMENT_MANAGER_MODEL = os.getenv("ENGAGEMENT_MANAGER_MODEL", "qwen3:8b")
MARKET_RESEARCHER_MODEL = os.getenv("MARKET_RESEARCHER_MODEL", "qwen3:14b")
STRATEGY_CONSULTANT_MODEL = os.getenv("STRATEGY_CONSULTANT_MODEL", "qwen3:32b")
EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "qwen3:14b")

AGENT_MODELS = {
    "engagement_manager": ENGAGEMENT_MANAGER_MODEL,
    "market_researcher": MARKET_RESEARCHER_MODEL,
    "strategy_consultant": STRATEGY_CONSULTANT_MODEL,
    "evaluator": EVALUATOR_MODEL,
}


# ---------------------------------------------------------------------------
# Utility — robust JSON extraction from LLM output
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling think-tags, code fences, and
    common LLM quirks (trailing commas, single quotes, etc.)."""
    # Strip <think>…</think> blocks (Qwen 3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json … ``` code fences
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)  # use the extracted block for repair below

    # Try finding the outermost { … }
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            # Attempt common repairs
            repaired = _repair_json(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from LLM response: {text[:300]}")


def _repair_json(text: str) -> str:
    """Fix common LLM JSON mistakes: trailing commas, unescaped newlines."""
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    # Replace single quotes used as string delimiters with double quotes
    # (only outside of already double-quoted strings — best-effort)
    # Remove control characters except \n \r \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def strip_think_tags(text: str) -> str:
    """Remove <think>…</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# System Prompts
# ---------------------------------------------------------------------------
ENGAGEMENT_MANAGER_PROMPT = """\
You are the Engagement Manager at an AI consulting firm. Your sole \
responsibility is to break down client business questions into a clear, \
actionable analysis plan.

RESPONSIBILITIES:
- Understand the client's core business question
- Decompose it into 3–5 focused workstreams
- For each workstream, define specific sub-questions that need investigation
- Create a structured analysis plan that guides the research team

RESTRICTIONS — You must obey these absolutely:
- Do NOT perform any market research, financial analysis, or risk assessment
- Do NOT write any part of the final consulting report
- Do NOT make strategic recommendations
- ONLY produce the analysis plan — nothing more

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "analysis_plan": {
    "business_question_summary": "A clear restatement of what the client is asking",
    "workstreams": [
      {
        "id": 1,
        "title": "Workstream title",
        "description": "What this workstream investigates",
        "key_questions": ["Specific question 1", "Specific question 2"]
      }
    ]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

MARKET_RESEARCHER_PROMPT = """\
You are a Market Researcher at an AI consulting firm. You investigate market \
landscapes and produce structured research findings.

RESPONSIBILITIES:
- Analyze the target market: size, growth trajectory, and dynamics
- Identify key competitors and their market positions
- Spot relevant market trends and emerging patterns
- Define customer segments and their characteristics
- Distill findings into clear, evidence-based insights

RESTRICTIONS — You must obey these absolutely:
- Do NOT perform financial analysis (costs, ROI, projections)
- Do NOT assess risks
- Do NOT write the final consulting report or strategic recommendations
- ONLY produce market research findings

You will receive the client question and an analysis plan with workstreams. \
Focus your research on addressing the relevant workstreams and key questions.

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "market_analysis": {
    "market_overview": "High-level overview of the relevant market",
    "market_size_and_growth": "Current size estimates and growth projections",
    "key_competitors": [
      {
        "name": "Competitor name",
        "description": "What they do and how they compete",
        "market_position": "Their standing (leader/challenger/niche)"
      }
    ],
    "market_trends": ["Trend 1", "Trend 2"],
    "customer_segments": [
      {
        "segment": "Segment name",
        "description": "Characteristics and needs",
        "size_estimate": "Relative or absolute size estimate"
      }
    ],
    "key_findings": ["Finding 1", "Finding 2", "Finding 3"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

STRATEGY_CONSULTANT_PROMPT = """\
You are a senior Strategy Consultant at an AI consulting firm. You synthesize \
research findings into a final consulting recommendation.

RESPONSIBILITIES:
- Analyze the market research findings you receive
- Develop 2–3 distinct strategic options with clear pros, cons, and tradeoffs
- Recommend one option with thorough justification
- Write a complete, professional consulting report

RESTRICTIONS — You must obey these absolutely:
- Do NOT search for new information — work only with what you receive
- Do NOT modify or contradict the research findings
- Base your analysis entirely on the provided market research and analysis plan

Write a professional consulting report in Markdown format with these sections:

# Executive Summary
Brief overview of the situation and your top-line recommendation.

## Situation Analysis
What the client is facing, based on the market research.

## Strategic Options
Present 2–3 options, each with:
- Description of the approach
- Pros and advantages
- Cons and risks
- Estimated effort/complexity

## Recommendation
Your recommended option with detailed justification.

## Implementation Roadmap
Phased action plan: short-term (0–3 months), mid-term (3–12 months), \
long-term (12+ months).

Write in a professional, concise consulting style. Use data from the market \
research to support your points. Be specific and actionable.\
"""

EVALUATOR_PROMPT = """\
You are an independent Evaluator at an AI consulting firm. Your job is to \
objectively assess the quality of consulting reports.

RESPONSIBILITIES:
- Score the report on 6 criteria (1–10 scale each)
- Provide honest, specific justification for each score
- Calculate an overall score (average of all six)
- Write a brief evaluation summary

RESTRICTIONS — You must obey these absolutely:
- Do NOT modify the report
- Do NOT add new analysis or recommendations
- ONLY evaluate and score — nothing more
- Be critical and honest — do not inflate scores

SCORING CRITERIA:
1. Completeness   — Are all aspects of the business question addressed?
2. Accuracy       — Are claims and numbers supported and realistic?
3. Coherence      — Does the analysis flow logically from data to recommendation?
4. Structure      — Is it well-organized like a professional consulting deliverable?
5. Actionability  — Are the recommendations specific enough to act on?
6. Critical Depth — Are risks, limitations, and counterarguments addressed?

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "evaluation": {
    "scores": {
      "completeness":   {"score": 8, "justification": "Specific reason..."},
      "accuracy":       {"score": 7, "justification": "Specific reason..."},
      "coherence":      {"score": 8, "justification": "Specific reason..."},
      "structure":      {"score": 9, "justification": "Specific reason..."},
      "actionability":  {"score": 7, "justification": "Specific reason..."},
      "critical_depth": {"score": 6, "justification": "Specific reason..."}
    },
    "overall_score": 7.5,
    "summary": "Brief overall assessment of 2–3 sentences"
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------
class EngagementManager:
    """Decomposes the client question into workstreams and an analysis plan."""

    name = "engagement_manager"
    display_name = "Engagement Manager"
    model = ENGAGEMENT_MANAGER_MODEL

    def run(self, question: str) -> dict:
        """Return the analysis plan as a parsed dict."""
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": ENGAGEMENT_MANAGER_PROMPT},
                {"role": "user", "content": f"Client question:\n{question}"},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class MarketResearcher:
    """Investigates the market landscape and produces structured findings."""

    name = "market_researcher"
    display_name = "Market Researcher"
    model = MARKET_RESEARCHER_MODEL

    def run(self, question: str, analysis_plan: dict) -> dict:
        """Return the market analysis as a parsed dict.

        Receives only the question and the analysis plan (tool restriction:
        cannot see any other agent's output).
        """
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            "Based on the analysis plan above, produce a comprehensive market "
            "analysis addressing the relevant workstreams and key questions."
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": MARKET_RESEARCHER_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class StrategyConsultant:
    """Synthesizes findings into a final consulting report (Markdown, streamed)."""

    name = "strategy_consultant"
    display_name = "Strategy Consultant"
    model = STRATEGY_CONSULTANT_MODEL

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
    ) -> Generator[str, None, None]:
        """Stream the consulting report token-by-token.

        Receives the question, analysis plan, and market research
        (tool restriction: cannot search for new data).
        """
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            "Using the analysis plan and market research above, write a "
            "complete consulting report with your strategic recommendation."
        )
        stream = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": STRATEGY_CONSULTANT_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            stream=True,
        )
        yield from _filter_think_stream(stream)


class Evaluator:
    """Independently scores the final consulting report on a rubric."""

    name = "evaluator"
    display_name = "Evaluator"
    model = EVALUATOR_MODEL

    def run(self, question: str, report: str) -> dict:
        """Return the evaluation scorecard as a parsed dict.

        Receives only the original question and the final report
        (tool restriction: cannot see intermediate agent outputs).
        """
        user_prompt = (
            f"ORIGINAL CLIENT QUESTION:\n{question}\n\n"
            f"CONSULTING REPORT TO EVALUATE:\n{report}\n\n"
            "Evaluate the above consulting report on all 6 criteria. "
            "Be critical and honest in your assessment."
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": EVALUATOR_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


# ---------------------------------------------------------------------------
# Streaming think-tag filter
# ---------------------------------------------------------------------------
def _filter_think_stream(stream) -> Generator[str, None, None]:
    """Filter <think>…</think> blocks from an Ollama streaming response."""
    in_think = False
    for chunk in stream:
        content = chunk["message"]["content"]
        if not content:
            continue

        if in_think:
            if "</think>" in content:
                _, _, after = content.partition("</think>")
                in_think = False
                if after:
                    yield after
        else:
            if "<think>" in content:
                before, _, remainder = content.partition("<think>")
                if before:
                    yield before
                if "</think>" in remainder:
                    _, _, after = remainder.partition("</think>")
                    if after:
                        yield after
                else:
                    in_think = True
            else:
                yield content
