"""
Level 4 — Hybrid Organizational Workflow (BIS).

Five specialized agents operating in a hierarchical hybrid structure:

  1. Engagement Manager  — decomposes AND reviews every intermediate output
  2. Market Researcher   — investigates the market landscape
  3. Financial Analyst   — handles all quantitative/financial analysis
  4. Risk Analyst        — identifies and assesses risks
  5. Strategy Consultant — synthesizes all inputs into a consulting report

(The Evaluator runs in a separate standalone app powered by Kimi 2.5
and is not part of this backend.)

Key differences from Level 3:
  - The EM acts as a Managing Partner: reviews intermediate outputs and can
    send work back with specific revision feedback (max 1 revision per agent).
  - All agents use /think or /no_think mode toggles for Qwen 3.
  - Stricter anti-hallucination prompts with mandatory source tracing.
  - Financial Analyst switched from DeepSeek R1 to Qwen 3 14B.
  - Strategy Consultant upgraded to Qwen 3.5 27B (256K context).

Models (from level4_models.md):
  Engagement Manager  -> qwen3:8b          (fast decomposition + review)
  Market Researcher   -> qwen3:14b         (broad knowledge, /no_think synthesis)
  Financial Analyst   -> qwen3:14b         (/think for math, /no_think for extraction)
  Risk Analyst        -> qwen3:14b         (/think for analytical reasoning)
  Strategy Consultant -> qwen3.5:27b       (256K context, superior writing)
"""

import json
import os
import re
from typing import Generator

import ollama
import requests
from bs4 import BeautifulSoup
from ddgs import DDGS

# ---------------------------------------------------------------------------
# Model configuration (override via environment variables)
# ---------------------------------------------------------------------------
ENGAGEMENT_MANAGER_MODEL = os.getenv("ENGAGEMENT_MANAGER_MODEL", "qwen3:8b")
MARKET_RESEARCHER_MODEL = os.getenv("MARKET_RESEARCHER_MODEL", "qwen3:14b")
FINANCIAL_ANALYST_MODEL = os.getenv("FINANCIAL_ANALYST_MODEL", "qwen3:14b")
RISK_ANALYST_MODEL = os.getenv("RISK_ANALYST_MODEL", "qwen3:14b")
STRATEGY_CONSULTANT_MODEL = os.getenv("STRATEGY_CONSULTANT_MODEL", "qwen3.5:27b")

AGENT_MODELS = {
    "engagement_manager": ENGAGEMENT_MANAGER_MODEL,
    "market_researcher": MARKET_RESEARCHER_MODEL,
    "financial_analyst": FINANCIAL_ANALYST_MODEL,
    "risk_analyst": RISK_ANALYST_MODEL,
    "strategy_consultant": STRATEGY_CONSULTANT_MODEL,
}


# ---------------------------------------------------------------------------
# Utility -- robust JSON extraction from LLM output
# ---------------------------------------------------------------------------
def extract_json(text: str) -> dict:
    """Parse JSON from LLM output, handling think-tags, code fences, and
    common LLM quirks (trailing commas, single quotes, etc.)."""
    # Strip <think>...</think> blocks (Qwen 3 thinking mode)
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = text.strip()

    # Try direct parse first
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # Try extracting from ```json ... ``` code fences
    m = re.search(r"```json\s*(.*?)\s*```", text, flags=re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except json.JSONDecodeError:
            text = m.group(1)  # use the extracted block for repair below

    # Try finding the outermost { ... }
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
    # Remove control characters except \n \r \t
    text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f]", "", text)
    return text


def strip_think_tags(text: str) -> str:
    """Remove <think>...</think> blocks from text."""
    return re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()


# ---------------------------------------------------------------------------
# Shared tools -- search_web and read_document (used by MR and RA)
# ---------------------------------------------------------------------------

_SEARCH_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "search_web",
            "description": (
                "Search the web for information about a topic. "
                "Returns titles, URLs, and text snippets for the top results."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query to run.",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return (default 5, max 10).",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "read_document",
            "description": (
                "Fetch and read the text content of a web page by URL. "
                "Use this after search_web to read the full content of a relevant result."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {
                        "type": "string",
                        "description": "The URL of the page to fetch.",
                    },
                },
                "required": ["url"],
            },
        },
    },
]

_HEADERS = {"User-Agent": "Mozilla/5.0 (compatible; ResearchBot/1.0)"}
_DOC_CHAR_LIMIT = 4000  # increased from L3's 3000 for deeper reading


def search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a DuckDuckGo text search and return structured results."""
    max_results = min(int(max_results), 10)
    print(f"[tool:search_web] query={query!r} max_results={max_results}", flush=True)
    results = []
    try:
        with DDGS() as ddgs:
            for r in ddgs.text(query, max_results=max_results):
                results.append({
                    "title": r.get("title", ""),
                    "url": r.get("href", ""),
                    "snippet": r.get("body", ""),
                })
    except Exception as exc:
        results.append({"error": str(exc)})
    print(f"[tool:search_web] returned {len(results)} results", flush=True)
    return results


def read_document(url: str) -> str:
    """Fetch a URL and return cleaned plain text, truncated to _DOC_CHAR_LIMIT chars."""
    print(f"[tool:read_document] url={url}", flush=True)
    try:
        resp = requests.get(url, timeout=10, headers=_HEADERS)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()
        text = soup.get_text(separator="\n", strip=True)
        print(f"[tool:read_document] fetched {len(text)} chars (truncated to {_DOC_CHAR_LIMIT})", flush=True)
        return text[:_DOC_CHAR_LIMIT]
    except Exception as exc:
        print(f"[tool:read_document] error: {exc}", flush=True)
        return f"[Error fetching {url}: {exc}]"


# ---------------------------------------------------------------------------
# Risk Analyst tools -- assess_risk (plus search_web / read_document above)
# ---------------------------------------------------------------------------

_RA_TOOLS = _SEARCH_TOOLS + [
    {
        "type": "function",
        "function": {
            "name": "assess_risk",
            "description": (
                "Structure a risk assessment for a specific identified risk. "
                "Takes a risk title and description, and returns a formatted "
                "risk entry with probability and impact ratings."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "title": {
                        "type": "string",
                        "description": "Short title of the risk.",
                    },
                    "description": {
                        "type": "string",
                        "description": "Detailed description of the risk.",
                    },
                    "category": {
                        "type": "string",
                        "enum": ["regulatory", "market", "operational", "competitive", "financial"],
                        "description": "Risk category.",
                    },
                    "probability": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Likelihood of the risk materializing.",
                    },
                    "impact": {
                        "type": "string",
                        "enum": ["low", "medium", "high"],
                        "description": "Severity of impact if the risk materializes.",
                    },
                },
                "required": ["title", "description", "category", "probability", "impact"],
            },
        },
    },
]


def assess_risk(
    title: str,
    description: str,
    category: str,
    probability: str,
    impact: str,
) -> dict:
    """Return a structured risk assessment entry."""
    print(f"[tool:assess_risk] title={title!r} cat={category} prob={probability} impact={impact}", flush=True)
    score_map = {"low": 1, "medium": 2, "high": 3}
    prob_score = score_map.get(probability, 2)
    impact_score = score_map.get(impact, 2)
    risk_score = prob_score * impact_score
    risk_level = "low" if risk_score <= 2 else "medium" if risk_score <= 4 else "high" if risk_score <= 6 else "critical"
    return {
        "title": title,
        "description": description,
        "category": category,
        "probability": probability,
        "impact": impact,
        "risk_score": risk_score,
        "risk_level": risk_level,
    }


# ---------------------------------------------------------------------------
# Tool dispatch
# ---------------------------------------------------------------------------

def _dispatch_tool(name: str, args: dict) -> str:
    """Execute a tool call by name and return its result as a JSON string."""
    if name == "search_web":
        result = search_web(**args)
    elif name == "read_document":
        result = read_document(**args)
    elif name == "assess_risk":
        result = assess_risk(**args)
    else:
        result = {"error": f"Unknown tool: {name}"}
    return json.dumps(result, ensure_ascii=False)


# ---------------------------------------------------------------------------
# System Prompts (Level 4 -- stricter anti-hallucination)
# ---------------------------------------------------------------------------
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
2. SOURCE GROUNDING: Are all claims backed by cited sources (URLs) or \
explicitly marked as assumptions? Flag any unsourced statistics or claims.
3. NO HALLUCINATIONS: Does the output contain any invented data, fabricated \
sources, or numbers that cannot be traced to the research? Flag specific \
examples.
4. CONSISTENCY: Does the output contradict any data from previous agents? \
Flag contradictions.
5. QUALITY: Is it professional, well-structured, and detailed enough?

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

MARKET_RESEARCHER_RESEARCH_PROMPT = """\
You are a Market Researcher at an AI consulting firm. Your task right now is \
to gather information using your research tools.

You have two tools available:
- search_web(query, max_results): search the web for relevant information
- read_document(url): fetch and read the full text of a web page

CRITICAL INSTRUCTIONS -- follow this workflow:
1. Start by calling search_web with a broad query about the market
2. IMMEDIATELY call read_document on the 2-3 most relevant URLs from the \
search results -- search snippets are NOT enough, you MUST read full pages
3. Then search for specific topics: competitors, market size, trends, etc.
4. For EACH search, call read_document on at least 1-2 of the top URLs
5. Continue until you have covered all workstreams in the analysis plan

ANTI-HALLUCINATION RULES:
- You MUST call read_document at least 4 times during your research
- Search snippets alone are too shallow -- they contain only 1-2 sentences \
and often have outdated or incomplete data
- The full page content from read_document is essential for accurate data
- Track every URL you read -- you will need to cite them in your synthesis
- If a search returns no useful results, say so -- do NOT make up data

RESTRICTIONS:
- Do NOT produce financial analysis, ROI estimates, or cost projections
- Do NOT assess risks
- Only gather market intelligence\
"""

MARKET_RESEARCHER_SYNTHESIS_PROMPT = """\
You are a Market Researcher at an AI consulting firm. You have already \
conducted web research and the findings are provided below. Now synthesize \
that research into structured findings.

ANTI-HALLUCINATION RULES -- these are ABSOLUTE:
- Every claim MUST have a source URL in parentheses after it
- Every statistic or number MUST include the URL where it was found
- If the research does NOT contain data for a field, you MUST write \
"Data not available from research" -- do NOT invent numbers
- Do NOT extrapolate beyond what the sources say
- Do NOT combine partial data from different sources to create new statistics
- If two sources give different numbers, report BOTH with their URLs

RESTRICTIONS -- You must obey these absolutely:
- Do NOT perform financial analysis (costs, ROI, projections)
- Do NOT assess risks
- Do NOT write the final consulting report or strategic recommendations
- Base your output ONLY on the research findings provided

OUTPUT FORMAT -- You must respond with valid JSON matching this exact structure:
{
  "market_analysis": {
    "market_overview": "High-level overview with (source URL) citations",
    "market_size_and_growth": "Size estimates and growth projections with (source URL)",
    "key_competitors": [
      {
        "name": "Competitor name",
        "description": "What they do and how they compete",
        "market_position": "Their standing (leader/challenger/niche)",
        "source": "URL where this information was found"
      }
    ],
    "market_trends": ["Trend 1 (source URL)", "Trend 2 (source URL)"],
    "customer_segments": [
      {
        "segment": "Segment name",
        "description": "Characteristics and needs",
        "size_estimate": "Relative or absolute size with (source URL)"
      }
    ],
    "key_findings": ["Finding 1 (source URL)", "Finding 2 (source URL)"],
    "sources": ["URL 1", "URL 2", "URL 3", "URL 4"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

FINANCIAL_ANALYST_PROMPT = """\
You are a Financial Analyst at an AI consulting firm. You perform all \
quantitative and financial analysis for the team.

RESPONSIBILITIES:
- Extract specific numbers from the market research data (market size, growth \
rates, competitor pricing, salary ranges, etc.) and use THOSE as your inputs
- Calculate cost estimates for the proposed business action
- Create revenue projections for 3 scenarios (conservative, moderate, aggressive)
- Calculate ROI and break-even timeline
- Perform sensitivity analysis on key assumptions
- Show ALL calculations step by step in your reasoning before outputting JSON

ANTI-HALLUCINATION RULES -- these are ABSOLUTE:
- ONLY use numbers that appear in the market research data or can be directly \
derived from it with simple arithmetic
- For EVERY number you use, state its source: "Based on market research: X"
- If the market research does NOT provide a specific number you need, you MUST:
  (a) State what number is missing
  (b) State your assumption explicitly
  (c) Mark it as "[ASSUMED]" in the output
  (d) Use conservative estimates for assumptions
- Show your math: write out every equation with actual numbers
- Cross-check: verify totals add up, break-even is consistent with revenue/costs
- NEVER invent market data, competitor revenue, or industry statistics
- If you cannot perform a calculation due to missing data, say so explicitly

RESTRICTIONS -- You must obey these absolutely:
- Do NOT perform market research -- that data is already provided
- Do NOT assess non-financial risks (only flag financial risks)
- Do NOT write the final consulting report or strategic recommendations
- Do NOT invent market data -- use only what is provided

OUTPUT FORMAT -- respond with valid JSON matching this structure:
{
  "financial_analysis": {
    "executive_summary": "Brief overview of the financial outlook",
    "data_inputs_used": "List the key numbers extracted from market research with their sources",
    "cost_estimates": [
      {
        "category": "Cost category name",
        "amount": "$X,XXX",
        "timeframe": "one-time / monthly / annual",
        "notes": "Key assumptions -- cite market research data or mark [ASSUMED]"
      }
    ],
    "revenue_projections": [
      {
        "scenario": "Conservative / Moderate / Aggressive",
        "year_1": "$X,XXX",
        "year_2": "$X,XXX",
        "year_3": "$X,XXX",
        "assumptions": "Key assumptions -- reference market research data"
      }
    ],
    "roi_analysis": "Expected ROI with calculation shown",
    "break_even_timeline": "When the investment breaks even, with math shown",
    "sensitivity_analysis": "How results change if key assumptions vary by +/-20%",
    "key_financial_risks": ["Financial risk 1", "Financial risk 2"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

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

ANTI-HALLUCINATION RULES:
- Every risk you identify MUST be based on information from search results or \
the provided market research / financial analysis data
- Cite the source (URL or "from market research" / "from financial analysis") \
for each risk
- Do NOT invent regulatory requirements or compliance rules -- verify via search
- If you cannot find evidence for a risk, do NOT include it

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

ANTI-HALLUCINATION RULES -- these are ABSOLUTE:
- Base your output ONLY on the risk research provided -- do not invent data
- Every risk must cite its source (URL, "market research data", or "financial data")
- Do NOT invent regulatory requirements or compliance rules
- Only identify risks -- do NOT propose full solutions, only brief mitigation ideas
- If the research did not cover a risk category, say "No risks identified in \
this category from available research" -- do NOT fill in gaps with guesses

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
        "source": "URL or data source where this risk was identified",
        "mitigation_suggestion": "Brief mitigation idea"
      }
    ],
    "key_risk_factors": ["Key factor 1", "Key factor 2"]
  }
}

Respond ONLY with the JSON object. No commentary, no markdown, no extra text.\
"""

STRATEGY_CONSULTANT_PROMPT = """\
You are a senior Strategy Consultant at an AI consulting firm. You synthesize \
all research inputs -- market research, financial analysis, AND risk assessment \
-- into a final consulting recommendation.

RESPONSIBILITIES:
- Analyze the market research, financial analysis, and risk assessment findings
- Develop 2-3 distinct strategic options with clear pros, cons, and tradeoffs
- Recommend one option with thorough justification
- Write a complete, professional consulting report

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


# ---------------------------------------------------------------------------
# Agent classes
# ---------------------------------------------------------------------------
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
                {"role": "user", "content": f"Client question:\n{question}\n\n/think"},
            ],
            format="json",
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
        )
        # Use /no_think for fast, direct review
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a quality reviewer. Respond ONLY with the requested JSON."},
                {"role": "user", "content": prompt + "\n\n/no_think"},
            ],
            format="json",
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


class MarketResearcher:
    """Investigates the market landscape using web search tools, then produces
    structured findings via a two-phase approach.

    Level 4 changes:
      - /no_think mode during synthesis to reduce hallucination
      - Stricter source citation requirements
      - Supports revision with EM feedback
    """

    name = "market_researcher"
    display_name = "Market Researcher"
    model = MARKET_RESEARCHER_MODEL
    MAX_TOOL_ROUNDS = 8

    def run(
        self,
        question: str,
        analysis_plan: dict,
        revision_feedback: str | None = None,
    ) -> dict:
        """Return the market analysis as a parsed dict.

        If revision_feedback is provided, does a targeted second research pass
        focusing on the gaps identified by the EM.
        """
        research_context = self._research_phase(question, analysis_plan, revision_feedback)
        return self._synthesis_phase(question, analysis_plan, research_context)

    def _research_phase(
        self,
        question: str,
        analysis_plan: dict,
        revision_feedback: str | None = None,
    ) -> str:
        """Run the tool loop and return a text summary of all gathered findings."""
        revision_instruction = ""
        if revision_feedback:
            revision_instruction = (
                f"\n\nREVISION REQUIRED -- The Engagement Manager reviewed your "
                f"previous output and found issues:\n{revision_feedback}\n\n"
                f"Focus your research on addressing these specific gaps. "
                f"Search for the missing data and read relevant pages."
            )

        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            "Use search_web and read_document to gather the market information "
            "needed to address every workstream in the analysis plan. "
            "You MUST call read_document at least 4 times on different URLs. "
            "When you have enough data, stop calling tools."
            f"{revision_instruction}"
        )
        messages = [
            {"role": "system", "content": MARKET_RESEARCHER_RESEARCH_PROMPT},
            {"role": "user", "content": user_prompt},
        ]

        tool_results: list[str] = []

        for _ in range(self.MAX_TOOL_ROUNDS):
            response = ollama.chat(
                model=self.model,
                messages=messages,
                tools=_SEARCH_TOOLS,
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
        self, question: str, analysis_plan: dict, research_context: str
    ) -> dict:
        """Convert the raw research into the required JSON schema.
        Uses /no_think to minimize hallucination during factual synthesis."""
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"RESEARCH FINDINGS:\n{research_context}\n\n"
            "Synthesize the research findings above into a structured market "
            "analysis. Every claim must cite a source URL from the research.\n\n"
            "/no_think"
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": MARKET_RESEARCHER_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class FinancialAnalyst:
    """Handles all quantitative/financial analysis.

    Level 4 changes:
      - Switched from DeepSeek R1 to Qwen 3 14B
      - Uses /think mode for calculations, /no_think for data extraction
      - Must explicitly trace every number to market research data
      - Supports revision with EM feedback
    """

    name = "financial_analyst"
    display_name = "Financial Analyst"
    model = FINANCIAL_ANALYST_MODEL

    def run(
        self,
        question: str,
        analysis_plan: dict,
        market_analysis: dict,
        revision_feedback: str | None = None,
    ) -> dict:
        """Return the financial analysis as a parsed dict."""
        revision_instruction = ""
        if revision_feedback:
            revision_instruction = (
                f"\n\nREVISION REQUIRED -- The Engagement Manager reviewed your "
                f"previous output and found issues:\n{revision_feedback}\n\n"
                f"Fix these specific issues. Ensure every number traces to the "
                f"market research data or is explicitly marked [ASSUMED]."
            )

        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            "STEP 1: Extract all quantitative data from the market research above "
            "(market size, growth rates, competitor revenue, pricing, salaries, etc.)\n\n"
            "STEP 2: Using ONLY those extracted numbers as inputs, calculate:\n"
            "- Cost estimates (setup, operations, marketing, staffing, technology)\n"
            "- Revenue projections for 3 scenarios (conservative, moderate, aggressive)\n"
            "- ROI and break-even timeline\n"
            "- Sensitivity analysis (what happens if key inputs change by +/-20%)\n\n"
            "STEP 3: Show all calculations step by step in your thinking, "
            "then output the final structured JSON.\n\n"
            "IMPORTANT: Every number in your output must be traceable to either "
            "the market research data or an explicitly stated [ASSUMED] assumption."
            f"{revision_instruction}\n\n"
            "/think"
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": FINANCIAL_ANALYST_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


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
        Uses /think for analytical reasoning about risk levels."""
        user_prompt = (
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            f"RISK RESEARCH FINDINGS:\n{research_context}\n\n"
            "Synthesize the risk research above into a structured risk assessment. "
            "Include all risks identified during the research phase. "
            "Cite the source for each risk.\n\n"
            "/think"
        )
        response = ollama.chat(
            model=self.model,
            messages=[
                {"role": "system", "content": RISK_ANALYST_SYNTHESIS_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
            format="json",
        )
        return extract_json(response["message"]["content"])


class StrategyConsultant:
    """Synthesizes all inputs into a final consulting report (Markdown, streamed).

    Level 4 changes:
      - Upgraded to Qwen 3.5 27B (256K context)
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
            f"CLIENT QUESTION:\n{question}\n\n"
            f"ANALYSIS PLAN:\n{json.dumps(analysis_plan, indent=2)}\n\n"
            f"MARKET RESEARCH FINDINGS:\n{json.dumps(market_analysis, indent=2)}\n\n"
            f"FINANCIAL ANALYSIS:\n{json.dumps(financial_analysis, indent=2)}\n\n"
            f"RISK ASSESSMENT:\n{json.dumps(risk_assessment, indent=2)}\n\n"
            "Using ALL the inputs above -- analysis plan, market research, "
            "financial analysis, and risk assessment -- write a complete "
            "consulting report with your strategic recommendation. "
            "Every claim must reference which research input it comes from."
            f"{revision_instruction}"
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


# ---------------------------------------------------------------------------
# Streaming think-tag filter
# ---------------------------------------------------------------------------
def _filter_think_stream(stream) -> Generator[str, None, None]:
    """Filter <think>...</think> blocks from an Ollama streaming response."""
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
