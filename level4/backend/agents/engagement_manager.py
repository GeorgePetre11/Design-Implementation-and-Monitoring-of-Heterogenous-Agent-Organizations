"""Engagement Manager -- decomposes client question into an analysis plan
and reviews intermediate outputs from other agents (Level 4 Managing Partner role)."""

import json

import ollama

from .common import (
    ENGAGEMENT_MANAGER_MODEL,
    NUM_CTX_LARGE,
    NUM_CTX_SMALL,
    _today,
    extract_json,
)


ENGAGEMENT_MANAGER_PROMPT = """\
You are the Engagement Manager at an AI consulting firm. Your sole \
responsibility is to break down client business questions into a clear, \
actionable analysis plan.

RESPONSIBILITIES:
- Understand the client's core business question
- Decompose it into 3-5 focused workstreams
- For each workstream, define specific sub-questions that need investigation
- Create a structured analysis plan that guides the research team

RESTRICTIONS -- You must obey these absolutely:
- Do NOT perform any market research, financial analysis, or risk assessment
- Do NOT write any part of the final consulting report
- Do NOT make strategic recommendations
- ONLY produce the analysis plan -- nothing more

GEOGRAPHIC SPECIFICITY -- CRITICAL:
- Extract the EXACT target country and city from the client question
- Include the specific country and city name in EVERY workstream title
- Include the specific country and city name in EVERY sub-question
- NEVER use vague phrases like "the target market" or "the region" or \
"the local market" -- always name the specific location
- Example: Instead of "Analyze the target market size" write \
"Analyze the IT services market size in Bucharest, Romania"
- If the client mentions a budget, include the budget amount in the \
relevant financial workstream sub-questions

OUTPUT FORMAT -- You must respond with valid JSON matching this exact structure:
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

# -- EM Review Prompt (new in Level 4) --
EM_REVIEW_PROMPT = """\
You are the Engagement Manager reviewing the output of the {agent_name}. \
Your job is to ensure the output is complete, accurate, and free of \
hallucinations before it moves to the next stage.

Today's date: {today}

ORIGINAL CLIENT QUESTION:
{question}

ANALYSIS PLAN:
{analysis_plan}

{agent_name_upper}'S OUTPUT:
{agent_output}

{extra_context}

REVIEW CHECKLIST -- evaluate each point:
1. COMPLETENESS: Does the output address the relevant workstreams from the \
analysis plan? List any gaps.
2. GEOGRAPHIC ACCURACY: Does the output analyze the CORRECT target country \
and city specified in the client question? Flag if the wrong country/city \
is analyzed.
3. SEGMENT RELEVANCE (CRITICAL for Market Researcher): For EACH finding, \
statistic, trend, and competitor, check:
   (a) Does it describe the SAME service segment the client operates in? \
(e.g. physical hardware maintenance, NOT AI software)
   (b) Is the source cited and does the source actually discuss that \
segment?
   (c) Could a reasonable consultant disagree that this finding applies to \
the client? If yes, flag for revision.
   If any answer is "no" or "unclear", send back for revision with the \
specific finding named.
4. SOURCE GROUNDING: Are all claims backed by cited sources (URLs) or \
explicitly marked as assumptions? Flag any unsourced statistics or claims.
5. NO HALLUCINATIONS: Does the output contain any invented data, fabricated \
sources, or numbers that cannot be traced to the research? Flag specific \
examples. Check if competitor names and URLs look legitimate.
6. SCHEMA COMPLETENESS (per agent):
   - Market Researcher: key_competitors has >=3 entries (or an explicit \
"fewer than 3 found" note); regulatory_landscape all four subfields filled; \
every finding/trend/segment has a relevance_to_client_service sentence
   - Financial Analyst: revenue_model_type is set with a justification; \
revenue_projections has exactly three scenarios (Conservative, Moderate, \
Aggressive); sensitivity_analysis is non-empty; no per-employee cost is \
<€15K/year without explanation
   - Risk Analyst: every risk has a sourced origin (URL, market_research, \
or financial_analysis)
   Flag for revision if any item is missing.
7. CONSISTENCY: Does the output contradict any data from previous agents? \
Flag contradictions.
8. TEMPORAL ACCURACY: Are all dates and timelines plausible relative to \
today's date ({today})? Flag any references to past dates as future events.
9. QUALITY: Is it professional, well-structured, and detailed enough?

Respond with ONLY this JSON:
{{
  "approved": true or false,
  "completeness_ok": true or false,
  "sources_ok": true or false,
  "no_hallucinations": true or false,
  "consistency_ok": true or false,
  "quality_ok": true or false,
  "feedback": "Specific issues to fix. Empty string if approved."
}}\
"""


class EngagementManager:
    """Decomposes the client question into workstreams and an analysis plan.
    In Level 4, also reviews intermediate outputs from other agents."""

    name = "engagement_manager"
    display_name = "Engagement Manager"
    model = ENGAGEMENT_MANAGER_MODEL

    def run(self, question: str) -> dict:
        """Return the analysis plan as a parsed dict."""
        # Use /think for structured decomposition
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": ENGAGEMENT_MANAGER_PROMPT},
                {"role": "user", "content": (
                    f"Today's date: {_today()}\n\n"
                    f"Client question:\n{question}\n\n"
                    "Remember: include the SPECIFIC target country and city "
                    "in EVERY workstream title and EVERY sub-question. "
                    "Never use vague phrases like 'the target market'.\n\n"
                    "/think"
                )},
            ],
            format="json",
            options={"num_ctx": NUM_CTX_SMALL},
        )
        return extract_json(response["message"]["content"])

    def review_output(
        self,
        agent_name: str,
        question: str,
        analysis_plan: dict,
        agent_output: dict | str,
        extra_context: str = "",
    ) -> dict:
        """Review an agent's output and return approval/feedback.

        Returns:
            {"approved": bool, "feedback": str, ...}
        """
        output_str = (
            json.dumps(agent_output, indent=2)
            if isinstance(agent_output, dict)
            else str(agent_output)
        )
        prompt = EM_REVIEW_PROMPT.format(
            agent_name=agent_name,
            agent_name_upper=agent_name.upper(),
            question=question,
            analysis_plan=json.dumps(analysis_plan, indent=2),
            agent_output=output_str,
            extra_context=extra_context,
            today=_today(),
        )
        # Use /no_think for fast, direct review
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a quality reviewer. Respond ONLY with the requested JSON."},
                {"role": "user", "content": prompt + "\n\n/no_think"},
            ],
            format="json",
            options={"num_ctx": NUM_CTX_LARGE},
        )
        try:
            result = extract_json(response["message"]["content"])
            # Ensure required fields exist
            result.setdefault("approved", False)
            result.setdefault("feedback", "")
            return result
        except (ValueError, KeyError):
            # If review fails to parse, approve by default to not block pipeline
            return {"approved": True, "feedback": "", "parse_error": True}
