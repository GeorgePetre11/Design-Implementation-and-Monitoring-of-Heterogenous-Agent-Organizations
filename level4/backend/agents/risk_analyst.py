"""Risk Analyst -- identifies and assesses risks using web search + assess_risk tool."""

import json

import ollama

from .common import (
    NUM_CTX_LARGE,
    RISK_ANALYST_MODEL,
    _extract_urls,
    extract_json,
)
from .tools import _RA_TOOLS, _dispatch_tool


RISK_ANALYST_RESEARCH_PROMPT = """\
You are a Risk Analyst at an AI consulting firm. Your task right now is to \
identify and assess risks related to the business question using your tools.

You have three tools available:
- search_web(query, max_results): search the web for risk-related information
- read_document(url): fetch and read the full text of a web page
- assess_risk(title, description, category, probability, impact): structure a \
risk assessment for a specific identified risk

INSTRUCTIONS:
- Use search_web to research regulatory, market, operational, and competitive risks
- Use read_document to get detailed information on specific risk factors
- Use assess_risk for each major risk you identify to create structured entries
- Identify at least 5-8 distinct risks across multiple categories

URL TRACKING RULES -- ABSOLUTE:
- After each search_web call, note the EXACT URLs returned
- When you call read_document, use the EXACT URL from search results
- Copy URLs character-by-character -- do NOT reconstruct from memory
- You will need to cite these exact URLs later

ANTI-HALLUCINATION RULES:
- Every risk you identify MUST be based on information from search results or \
the provided market research / financial analysis data
- Cite the source (URL or "from market research" / "from financial analysis") \
for each risk
- Do NOT invent regulatory requirements or compliance rules -- verify via search
- If you cannot find evidence for a risk, do NOT include it
- Do NOT add risks from your training knowledge -- only from research findings

RESTRICTIONS:
- Do NOT perform market research beyond what's needed for risk identification
- Do NOT do financial analysis
- Do NOT write strategic recommendations
- Only identify and assess risks -- do NOT propose full solutions\
"""

RISK_ANALYST_SYNTHESIS_PROMPT = """\
You are a Risk Analyst at an AI consulting firm. You have already researched \
and assessed risks. The findings are provided below. Now synthesize them into \
a structured risk assessment.

URL CITATION RULES -- these are ABSOLUTE:
- You will be given a URL REGISTRY containing every URL found during research
- You may ONLY cite URLs that appear in the URL REGISTRY
- COPY URLs exactly -- do NOT reconstruct or fabricate URLs
- If a risk came from market research or financial data (not a URL), write \
"source: market research data" or "source: financial analysis data"
- A fake URL is worse than no URL -- when in doubt, cite the data source \
instead of guessing a URL

ANTI-HALLUCINATION RULES -- these are ABSOLUTE:
- Base your output ONLY on the risk research provided -- do not invent data
- Do NOT invent regulatory requirements or compliance rules -- only cite \
regulations you found in actual search results
- Only identify risks -- do NOT propose full solutions, only brief mitigation ideas
- If the research did not cover a risk category, say "No risks identified in \
this category from available research" -- do NOT fill in gaps with guesses
- Do NOT add risks from your training knowledge that were not found in research

RESTRICTIONS -- You must obey these absolutely:
- Do NOT perform market research
- Do NOT do financial analysis
- Do NOT write the final consulting report or strategic recommendations

OUTPUT FORMAT -- respond with valid JSON matching this structure:
{
  "risk_assessment": {
    "overall_risk_level": "low / medium / high / critical",
    "risk_summary": "Brief overview of the risk landscape",
    "risks": [
      {
        "id": 1,
        "title": "Risk title",
        "description": "Detailed risk description",
        "category": "regulatory / market / operational / competitive / financial",
        "probability": "low / medium / high",
        "impact": "low / medium / high",
        "source": "EXACT URL from URL REGISTRY, or 'market research data' / 'financial analysis data'",
        "mitigation_suggestion": "Brief mitigation idea"
      }
    ],
    "key_risk_factors": ["Key factor 1", "Key factor 2"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""


class RiskAnalyst:
    """Identifies and assesses risks using web search and risk assessment tools.

    Level 4 changes:
      - Uses /think mode for analytical reasoning
      - Must cite sources for every risk
      - Cross-checks against financial analysis data
      - Supports revision with EM feedback
    """

    name = "risk_analyst"
    display_name = "Risk Analyst"
    model = RISK_ANALYST_MODEL
    MAX_TOOL_ROUNDS = 8

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
        revision_feedback: str | None = None,
    ) -> dict:
        """Return the risk assessment as a parsed dict."""
        research_context = self._research_phase(
            question, analysis_plan, market_analysis, financial_analysis, revision_feedback
        )
        return self._synthesis_phase(
            question, analysis_plan, market_analysis, financial_analysis, research_context
        )

    def _research_phase(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
        revision_feedback: str | None = None,
    ) -> str:
        """Run the risk research tool loop and return all findings."""
        revision_instruction = ""
        if revision_feedback:
            revision_instruction = (
                f"\n\nREVISION REQUIRED -- The Engagement Manager reviewed your "
                f"previous output and found issues:\n{revision_feedback}\n\n"
                f"Focus your research on addressing these specific gaps."
            )

        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            "Using the information above, identify and assess risks:\n"
            "1. Search for regulatory risks relevant to this business action\n"
            "2. Identify market risks based on the competitive landscape\n"
            "3. Consider operational risks of execution\n"
            "4. Assess competitive threats\n"
            "5. Flag any financial risks from the financial analysis\n"
            "6. Check if the financial projections are realistic given the risks\n\n"
            "Use search_web to research risks, read_document for details, "
            "and assess_risk to structure each identified risk. "
            "Identify at least 5 risks across multiple categories. "
            "When done, stop calling tools."
            f"{revision_instruction}"
        )
        messages = [
            {"role": "system", "content": RISK_ANALYST_RESEARCH_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        tool_results: list[str] = []

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=_RA_TOOLS,
                options={"num_ctx": NUM_CTX_LARGE},
            )
            msg = response["message"]
            messages.append(msg)

            tool_calls = msg.get("tool_calls") or []
            if not tool_calls:
                break

            for tc in tool_calls:
                name = tc["function"]["name"]
                args = tc["function"]["arguments"]
                result_str = _dispatch_tool(name, args)
                tool_results.append(f"[{name}({args})]\n{result_str}")
                messages.append({"role": "tool", "content": result_str})

        return "\n\n---\n\n".join(tool_results) if tool_results else "(no tool results)"

    def _synthesis_phase(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        financial_analysis: dict,
        research_context: str,
    ) -> dict:
        """Convert the risk research into the required JSON schema.
        Uses /think for analytical reasoning about risk levels.
        Includes a URL registry so the agent can only cite real URLs."""
        urls = _extract_urls(research_context)
        url_list = "\n".join(
            f"  [{i+1}] {u}" for i, u in enumerate(urls)
        ) if urls else "  (no URLs found in research)"

        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            f"RISK RESEARCH FINDINGS:\n{research_context}\n\n"
            f"URL REGISTRY — you may ONLY cite URLs from this list:\n{url_list}\n\n"
            "Synthesize the risk research above into a structured risk assessment. "
            "Include all risks identified during the research phase. "
            "For each risk, cite the source: use an EXACT URL from the URL REGISTRY, "
            "or 'market research data' / 'financial analysis data'. "
            "Do NOT fabricate or reconstruct URLs.\n\n"
            "/think"
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": RISK_ANALYST_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
            options={"num_ctx": NUM_CTX_LARGE},
        )
        return extract_json(response["message"]["content"])
