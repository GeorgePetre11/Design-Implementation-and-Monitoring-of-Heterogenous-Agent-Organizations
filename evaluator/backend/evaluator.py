"""
Evaluator Agent — Independent quality judge for consulting reports.

Uses DeepSeek-R1:70b for deep reasoning via chain-of-thought.
The model thinks through each criterion in <think> tags before
producing the final structured JSON scorecard.

This agent is intentionally standalone — it does NOT participate in
the pipeline. It receives a finished report and scores it independently.

Constraint enforcement:
  - System prompt: defines the rubric and restrictions (soft)
  - Output schema: must return valid EvaluationScorecard JSON (hard)
  - Tool restrictions: NO tools — cannot search, cannot modify the report
  - Data restriction: only sees the question and the final report
"""

import json
import os
import re

import ollama

EVALUATOR_MODEL = os.getenv("EVALUATOR_MODEL", "deepseek-r1:70b")

# ---------------------------------------------------------------------------
# System Prompt
# ---------------------------------------------------------------------------

EVALUATOR_SYSTEM_PROMPT = """\
You are an independent Evaluator at an AI consulting firm. Your sole \
responsibility is to critically assess the quality of consulting reports \
produced by the firm's analyst teams.

You are the FINAL quality gate. Your evaluation must be honest, rigorous, \
and well-justified. Do not be generous — clients pay for excellence.

EVALUATION RUBRIC — Score each criterion from 1 (very poor) to 10 (excellent):

1. COMPLETENESS (weight: 20%)
   - Are ALL aspects of the business question addressed?
   - Does the report cover market analysis, financial projections, risk \
assessment, strategic options, and implementation?
   - Are there gaps or missing workstreams?
   Score guide: 1-3 = major gaps, 4-6 = some areas missing or shallow, \
7-8 = thorough with minor gaps, 9-10 = comprehensive coverage of all aspects.

2. ACCURACY (weight: 20%)
   - Are claims supported by data, sources, or sound reasoning?
   - Are financial numbers realistic and internally consistent?
   - Are market claims plausible and specific (not vague)?
   Score guide: 1-3 = many unsupported or wrong claims, 4-6 = mix of \
supported and unsupported, 7-8 = mostly well-supported, 9-10 = rigorous \
data-driven analysis with clear sources.

3. COHERENCE (weight: 15%)
   - Does the analysis flow logically from data to conclusions to \
recommendations?
   - Are the strategic options consistent with the findings?
   - Do different sections contradict each other?
   Score guide: 1-3 = disjointed or contradictory, 4-6 = mostly logical \
but some jumps, 7-8 = clear logical flow, 9-10 = seamless argumentation.

4. STRUCTURE (weight: 15%)
   - Is the report well-organized with clear sections?
   - Does it follow professional consulting report conventions?
   - Is the formatting clean and easy to navigate?
   Score guide: 1-3 = disorganized, 4-6 = basic structure but rough, \
7-8 = well-organized and professional, 9-10 = publication-quality structure.

5. ACTIONABILITY (weight: 15%)
   - Are recommendations specific enough to act on?
   - Is there a clear implementation roadmap with phases/timelines?
   - Can a decision-maker use this report to make a real decision?
   Score guide: 1-3 = vague platitudes, 4-6 = some specific advice, \
7-8 = clear actionable recommendations, 9-10 = detailed roadmap with \
concrete next steps and timelines.

6. CRITICAL DEPTH (weight: 15%)
   - Are risks, limitations, and counterarguments addressed?
   - Does the report acknowledge uncertainty and tradeoffs?
   - Are multiple strategic options genuinely compared (not a strawman)?
   Score guide: 1-3 = no critical analysis, 4-6 = superficial risk \
mentions, 7-8 = thoughtful risk/tradeoff analysis, 9-10 = sophisticated \
critical thinking with nuanced tradeoffs.

RESTRICTIONS — You must obey these absolutely:
- Do NOT modify the report in any way
- Do NOT provide an improved version of the report
- Do NOT add new analysis or research
- ONLY evaluate and score — that is your entire job
- Be CALIBRATED: a score of 7 means genuinely good, not "average"
- A Level 1 (single agent, no tools) report SHOULD score lower than a \
Level 3 (specialized agents with tools) report if the quality differs

OVERALL SCORE CALCULATION:
overall_score = (completeness * 0.20) + (accuracy * 0.20) + \
(coherence * 0.15) + (structure * 0.15) + (actionability * 0.15) + \
(critical_depth * 0.15)

OUTPUT FORMAT — You must respond with valid JSON matching this exact structure:
{
  "completeness": {
    "score": 7,
    "justification": "The report covers market analysis and strategic options \
but lacks detailed financial projections..."
  },
  "accuracy": {
    "score": 6,
    "justification": "Market size claims are supported by sources, but \
revenue projections lack clear assumptions..."
  },
  "coherence": {
    "score": 8,
    "justification": "..."
  },
  "structure": {
    "score": 7,
    "justification": "..."
  },
  "actionability": {
    "score": 5,
    "justification": "..."
  },
  "critical_depth": {
    "score": 6,
    "justification": "..."
  },
  "overall_score": 6.55,
  "summary": "Overall assessment of the report in 2-3 sentences.",
  "strengths": [
    "Strength 1",
    "Strength 2",
    "Strength 3"
  ],
  "weaknesses": [
    "Weakness 1",
    "Weakness 2",
    "Weakness 3"
  ]
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""


# ---------------------------------------------------------------------------
# Utility — JSON extraction (same pattern as other levels)
# ---------------------------------------------------------------------------

def extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling think-tags, code fences, etc."""
    # Strip <think>...</think> blocks (DeepSeek R1 reasoning)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Direct parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Code fence extraction
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)

    # Outermost { ... }
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if m:
        candidate = m.group(0)
        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = _repair_json(candidate)
            try:
                return json.loads(repaired)
            except json.JSONDecodeError:
                pass

    raise ValueError(f"Could not extract JSON from evaluator response: {text[:300]}")


def _repair_json(text: str) -> str:
    """Fix common LLM JSON mistakes."""
    text = re.sub(r",\s*([}\]])", r"\1", text)
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


# ---------------------------------------------------------------------------
# Evaluator Agent
# ---------------------------------------------------------------------------

WEIGHTS = {
    "completeness": 0.20,
    "accuracy": 0.20,
    "coherence": 0.15,
    "structure": 0.15,
    "actionability": 0.15,
    "critical_depth": 0.15,
}

MAX_RETRIES = 3


class Evaluator:
    """Independent quality judge. Scores a consulting report against
    the six-criterion rubric using DeepSeek-R1:70b."""

    name = "evaluator"
    display_name = "Evaluator"
    model = EVALUATOR_MODEL

    def run(self, question: str, report: str, level: int) -> dict:
        """Evaluate the report and return a parsed scorecard dict.

        Args:
            question: The original client business question.
            report: The full consulting report (Markdown).
            level: Which complexity level produced this report (1-4).

        Returns:
            Parsed EvaluationScorecard as a dict.
        """
        user_prompt = (
            f"COMPLEXITY LEVEL: Level {level}\n"
            f"(Level 1 = single generic agent, no tools. "
            f"Level 2 = 3 specialized agents. "
            f"Level 3 = 5 specialized agents with web search and financial tools. "
            f"Level 4 = 6 agents with organizational workflows.)\n\n"
            f"ORIGINAL CLIENT QUESTION:\n{question}\n\n"
            f"CONSULTING REPORT TO EVALUATE:\n"
            f"{'=' * 60}\n"
            f"{report}\n"
            f"{'=' * 60}\n\n"
            f"Evaluate this Level {level} consulting report against all six "
            f"criteria in your rubric. Be rigorous and honest. Justify every score."
        )

        last_err = None
        for attempt in range(MAX_RETRIES):
            try:
                print(
                    f"[evaluator] attempt {attempt + 1}/{MAX_RETRIES} "
                    f"using model={self.model}",
                    flush=True,
                )
                response = ollama.chat(
                    model=self.model,
                    messages=[
                        {"role": "system", "content": EVALUATOR_SYSTEM_PROMPT},
                        {"role": "user", "content": user_prompt},
                    ],
                    format="json",
                )
                raw = response["message"]["content"]
                scorecard = extract_json(raw)

                # Recalculate overall_score to ensure correctness
                scorecard["overall_score"] = round(
                    sum(
                        scorecard[criterion]["score"] * weight
                        for criterion, weight in WEIGHTS.items()
                    ),
                    2,
                )

                return scorecard

            except Exception as e:
                last_err = e
                print(f"[evaluator] attempt {attempt + 1} failed: {e}", flush=True)
                if attempt < MAX_RETRIES - 1:
                    continue

        raise RuntimeError(
            f"Evaluator failed after {MAX_RETRIES} attempts. Last error: {last_err}"
        )
